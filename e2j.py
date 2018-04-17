import sys
import collections
import copy
import numpy as np
import argparse
import time
import random
import math
import io
# import codecs

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class EncoderDecoder(chainer.Chain):
    def __init__(self, encVocab, decVocab, args):
        super(EncoderDecoder, self).__init__()
        with self.init_scope():
            self.encEmbed = L.EmbedID(len(encVocab), args.embedDim)
            self.decEmbed = L.EmbedID(len(decVocab), args.embedDim)
            self.decOut = L.Linear(args.hiddenDim, len(decVocab))
            self.encNStepLSTM = L.NStepLSTM(1, args.embedDim, args.hiddenDim, args.dropout_rate)
            self.decLSTM = L.LSTM(args.embedDim + args.hiddenDim, args.hiddenDim)

            if args.attention:
                self.attnIn = L.Linear(args.hiddenDim, args.hiddenDim, nobias=True)
                self.attnOut = L.Linear(args.hiddenDim+args.hiddenDim, args.hiddenDim, nobias=True)

    def trainBatch(self, encSents, decSents, args):
        """main training"""
        ###encoder step
        encEmbed = self.getEmbeddings(encSents, args)
        hy, cy, ys = self.encNStepLSTM(hx=None, cx=None, xs=encEmbed)
        encOut = F.pad_sequence(ys) #[batch, max(sentlen), Dim]

        ###decoder step
        decEmbed = self.getEmbeddings(decSents, args) #embed のリストの状態
        decEmbed = F.pad_sequence(decEmbed).transpose([1, 0, 2]) #padding して[sentLen, batch, Dim]に変更

        decode_step = len(decEmbed) - 1
        decoderOutList = [0] * decode_step
        lstmStateList = [0] * decode_step
        firstInput = chainer.Variable(xp.zeros(hy[0].shape, dtype=xp.float32)) #decLSTMの最初に入力として初期化 embedじゃない方
        for i in range(decode_step):
            #decEmbedじゃないdecLSTMへの入力の準備と、decLSTMのstate準備
            if i == 0: #デコーダの最初のステップ
                self.set_state([cy[0], hy[0]])
                anoInput = firstInput
            else:
                self.set_state(lstmStateList[i - 1])
                anoInput = decoderOutList[i - 1]
            hOut = self.decLSTM(F.concat([decEmbed[i], anoInput], 1)) #decoder LSTMの出力
            lstmStateList[i] = self.get_state()
            decoderOutList[i] = self.attention(hOut, encOut, args) #decoder LSTMの出力をアテンションしたもの decoderの出力 decode_step * [batch, Dim]

        total_loss = chainer.Variable(xp.zeros((), dtype=xp.float32))
        proc = 0
        correct = 0
        incorrect = 0
        ###output層
        correctLabels = F.pad_sequence(decSents, padding=-1).T.array #TODO 何か嫌だから上手く書きたい　ってかこの関数全体何か汚い -1でパディングしたから1足したら0がeosトークンになるんじゃね？
        for i in range(decode_step):
            oVector = self.decOut(F.dropout(decoderOutList[i], args.dropout_rate))
            correctLabel = correctLabels[i + 1]

            proc += (xp.count_nonzero(correctLabel + 1)) ###TODO 0を数えてたらunkトークンがなくなるし、1足したら全部1以上になるンゴ
            # 必ずminibatchsizeでわる
            closs = F.softmax_cross_entropy(
                oVector, correctLabel, normalize=False) #normalize=Falseの意味？ paddingしてるからっぽい
            # これで正規化なしのloss  cf. seq2seq-attn code
            #total_loss_val += closs.data * cMBSize
            #if train_mode > 0:  # 学習データのみ backward する
            total_loss += closs
            # 実際の正解数を獲得したい
            t_correct = 0
            t_incorrect = 0
            # Devのときは必ず評価，学習データのときはオプションに従って評価
            # if train_mode == 0 or args.doEvalAcc > 0:
            # 予測した単語のID配列 CuPy
            pred_arr = oVector.data.argmax(axis=1)
            # 正解と予測が同じなら0になるはず
            # => 正解したところは0なので，全体から引く ###xp.count_nonzero()は間違えた数？
            t_correct = (correctLabel.size -
                         xp.count_nonzero(correctLabel - pred_arr)) #t_correct正解した数
            # 予測不要の数から正解した数を引く # +1はbroadcast
            t_incorrect = xp.count_nonzero(correctLabel + 1) - t_correct #xp.count_nonzero()は予測する必要のある数 つまりt_incorrectは間違えた数
            correct += t_correct
            incorrect += t_incorrect
        ####
        #total_loss.backward()

        return total_loss, (correct, incorrect, decode_step, proc)
            
    def attention(self, hOut, encOut, args): #TODO この関数も何か汚い
        """calc attention"""
        if args.attention == 0: #アテンションをしない時hOutをそのまま返す
            return hOut
        #ターゲット側の下準備
        hOut1 = self.attnIn(hOut) #[batch, Dim]
        hOut2 = F.expand_dims(hOut1, axis=1) #[batch, 1, Dim]
        hOut3 = F.broadcast_to(hOut2, (len(hOut2), len(encOut[0]), args.hiddenDim)) #[batch, max(enc_sentlen), Dim] 今encOutとhOut3は同じshapeのはず

        aval = F.sum(encOut*hOut3, axis=2) #[batch, sentlen]

        cAttn1 = F.softmax(aval) #[batch, max(enc_sentlen)] paddingで0のところはかなり小さい数字の確率で出てくる
        cAttn2 = F.expand_dims(cAttn1, axis=1) #[batch, 1, max(enc_sentlen)]
        cAttn3 = F.batch_matmul(cAttn2, encOut) #[batch, 1, Dim]
        context = F.reshape(cAttn3, (len(encOut), len(encOut[0][0]))) #[batch, Dim] エンコーダコンテキストベクトルの完成

        c1 = F.concat((hOut, context)) #[batch, Dim + Dim]
        c2 = self.attnOut(c1) #[bathc, Dim]
        return F.tanh(c2) #活性化
        
    
    def getEmbeddings(self, sentenceList, args):
        sentenceLen = xp.array([len(sentence) for sentence in sentenceList])
        sentenceSection = xp.cumsum(sentenceLen[:-1]).tolist()
        sentenceEmbed = F.dropout(self.encEmbed(F.concat(sentenceList, axis=0)), args.dropout_rate)
        return F.split_axis(sentenceEmbed, sentenceSection, axis=0, force_tuple=True)

    def set_state(self, state):
        self.decLSTM.set_state(state[0], state[1])

    def get_state(self):
        return [self.decLSTM.c, self.decLSTM.h]
    

class PrepareData:
    def __init__(self, args):
        """tokuninai"""
        self.unknown = "omoitukanai"

    def makeDict(self, vocabFile):
        """make dict"""
        d = {}
        d.setdefault("<unk>", len(d))  # 0番目 未知語 テストとかで出てきたらこれを置く。特に学習はいたしません
        d.setdefault("<bos>", len(d))  # 1番目 文頭のシンボル
        d.setdefault("<eos>", len(d))  # 2番目 文末のシンボル
        for line in open(vocabFile, "r"):
            for word in line.strip().split():
                if word == "<unk>":
                    continue
                elif word == "<s>":
                    continue
                elif word == "</s>":
                    continue
                d.setdefault(word, len(d))
        return d

    def sentenceListChange(self, sentenceList, wordDict):
        """id/word list to word/id list"""
        return [wordDict[word] if word in wordDict else wordDict["<unk>"] for word in sentenceList] ####辞書にあればwordIDなければunkを返す

    def makeSentenceList(self, fileName, wordDict):
        """rawfile to idList"""
        sentList = []
        for line in open(fileName, "r"):
            line = line.strip().split()
            indexList = self.sentenceListChange(line, wordDict)
            sentList.append([wordDict["<bos>"]] + indexList + [wordDict["<eos>"]])
        return sentList
        
    def makeInput(self, encSentList, decSentList):
        """make input for model"""
        return [(eS, dS) for eS, dS in zip(encSentList, decSentList)]

    
class EncoderDecoderUpdater(training.StandardUpdater):
    """model updater"""
    def __init__(self, train_iter, optimizer, args):
        super(EncoderDecoderUpdater, self).__init__(
            train_iter,
            optimizer,
        )

    def update_core(self):
        train_iter = self.get_iterator("main")
        optimizer = self.get_optimizer("main")

        batch = train_iter.__next__()

        encSents = [xp.array(x[0], dtype=xp.int32) for x in batch]
        decSents = [xp.array(x[1], dtype=xp.int32) for x in batch]

        loss, _ = optimizer.target.trainBatch(encSents, decSents, args)

        optimizer.target.cleargrads()
        loss.backward()
        optimizer.update()


def updateBeamThreshold__2(queue, input):
    ####これ早いの？　普通にlambda使ってソートすれば良くない？　そこまで差がでるんかな
    # list内の要素はlist,タプル，かつ，0番目の要素はスコアを仮定
    if len(queue) == 0:
        queue.append(input)
    else:
        # TODO 線形探索なのは面倒なので 効率を上げるためには要修正
        for i in range(len(queue)):
            if queue[i][0] <= input[0]:
                continue
            tmp = queue[i]
            queue[i] = input
            input = tmp
    return queue


def BeamDecording(model, encSent, decDict, args): #1文ずつ処理していく　beamの時にbatch処理になるため、
    """docording by beam search"""
    # train_mode = 0  # 評価なので
    # エンコードゾーン
    encEmbed = model.getEmbeddings([encSent], args) #list[1, array[sent, Dim]]
    hy, cy, ys = model.encNStepLSTM(hx=None, cx=None, xs=encEmbed)
    encOut = F.pad_sequence(ys) #[batch(1), max(sentlen), Dim]

    idx_bos = decDict["<bos>"]
    idx_eos = decDict["<eos>"]
    
    anoInput = chainer.Variable(xp.zeros(hy[0].shape, dtype=xp.float32)) #anoInputで最初の入力
    beam = [(0, [idx_bos], idx_bos, cy[0], hy[0], anoInput)] #beam_searchの最初のagenda agenda内(スコア, 予測単語列, 1つ前の予測単語, 前のLSTMstatecy, hy, encLSTMに入るもうひとつの入力)
    empty = (1.0e+100, [idx_bos], idx_bos, None, None, None) #新しいbeamを作る時の空agenda


    beam_size = args.beam_size
    for i in range(args.decode_len + 1):
        newBeam = [empty] * beam_size #次のbeamになる候補たち、最初は空agenda

        batch_size = len(beam) #batch_sizeは最初異なるから
        encOut = F.broadcast_to(encOut, (batch_size, len(encOut[0]), args.hiddenDim)) #[beam(batch), encsentlen, Dim] 入力文は1つだけどbeamをバッチとして考える エンコーダの出力をbatchだけ伸ばす
        zipbeam= list(zip(*beam)) #転地

        lstm_cy = F.concat(zipbeam[3], axis=0) #(スコア, 予測単語列, 1つ前の予測単語, 前のLSTMstate_cy, hy, encLSTMに入るもうひとつの入力) 前のcyをコンカット
        lstm_hy = F.concat(zipbeam[4], axis=0) #hyのconcat
        model.set_state([lstm_cy, lstm_hy]) #一つ前のbeamに入ってるlstmstateをセッティング
        # concat(a, axis=0) == vstack(a)
        anoInput = F.concat(zipbeam[5], axis=0) #anoInputのconcat [batch, Dim]
        # 一つ前の予測単語からdecoderの入力取得
        wordIndex = xp.array(zipbeam[2], dtype=np.int32) #[batch]
        decInput = model.getEmbeddings([wordIndex], args)[0] #[batch, Dim]]

        hOut = model.decLSTM(F.concat([decInput, anoInput], 1)) #decoder LSTMの出力
        lstmState = model.get_state()
        decOuts = model.attention(hOut, encOut, args) #decoder LSTMの出力をアテンションしたもの decoderの出力 decode_step * [batch, Dim]
        oVector = model.decOut(decOuts)

        nextWordProb = -F.log_softmax(oVector.data).data #予測単語

        nextWordProb[:, idx_bos] = 1.0e+100
        sortedIndex = xp.argsort(nextWordProb)[:, :beam_size] #[beam(batch), beam] beam毎の各要素に対してbeam個の候補を用意してる

        #agenda内(スコア, 予測単語列, 1つ前の予測単語, 前のLSTMstatecy, hy, encLSTMに入るもうひとつの入力)
        for z, b in enumerate(beam):
            if b[2] == idx_eos: # 修了単語
                newBeam = updateBeamThreshold__2(newBeam, b)
                continue

            if i != args.decode_len and b[0] > newBeam[-1][0]: ####まだ最後じゃない　かつ　beam内のやつのほうがしょぼいならここで修了 newに入れてあげる価値無し
                continue
            # 3
            # 次のbeamを作るために準備
            cy = lstmState[0][z:z+1, ] #こうやって書くとexpand_dimしなくていい？
            hy = lstmState[1][z:z+1, ]
            next_ano = decOuts[z:z+1, ]
            # 長さ制約的にEOSを選ばなくてはいけないという場合
            if i == args.decode_len:
                wordIndex = idx_eos
                newProb = nextWordProb[z][wordIndex] + b[0] ####スコアに長さ制約の関係でeosシンボルスコアがたされる
                nb = (newProb, b[1][:] + [wordIndex], wordIndex,
                      cy, hy, next_ano) ####beam内の今見てたbを更新してnbとした
                newBeam = updateBeamThreshold__2(newBeam, nb) ####nbがnewBeamに入れるかどうかの試験
                continue
            # 3
            # ここまでたどり着いたら最大beam個評価する
            # 基本的に sortedIndex_a[z] は len(beam) 個しかない
            for wordIndex in sortedIndex[z]: ####sortedIndex_a [beam, beam(batch)] のfor z, b in enumerate に入る前に作ってたencoderの予測高いtopBのやつ
                newProb = nextWordProb[z][wordIndex] + b[0]
                #return (newBeam, newProb, nextWordProb, sortedIndex)
                # import pdb; pdb.set_trace()
                if newProb > newBeam[-1][0]: ####newbeam の中の最低スコアより低いならばここでおしまいだ
                    continue
                    # break
                # ここまでたどり着いたら入れる
                nb = (newProb, b[1][:] + [wordIndex.tolist()], wordIndex,
                      cy, hy, next_ano) #(スコア, 予測単語列, 1つ前の予測単語, 前のLSTMstate_cy, hy, encLSTMに入るもうひとつの入力)
                newBeam = updateBeamThreshold__2(newBeam, nb) ####また判定してるんごおおおおお
                #####
        ################
        # 一時刻分の処理が終わったら，入れ替える
        beam = newBeam
        if all([True if b[2] == idx_eos else False for b in beam]): ####beam内の全ての前の単語がeosになったら終わろうか
            break
        # 次の入力へ
    beam = [(b[0], b[1]) for b in beam] ####beam内いろいろとindex単語列を単語列に変換したものに変更
    ####多分　beam内の各要素説明(謎スコア, 予測単語列, 一個前の予測単語, decLSTMに渡すlstmのstate, decLSTMに入るinputEmbじゃない方(attしてるやつ), フィルター(出力単語を一部制御))
    return beam


def rerankingByLengthNormalizedLoss(beam, wposi):
    beam.sort(key=lambda b: b[0] / (len(b[wposi]) - 1))
    return beam

# BeamDecording(model, encSent, decDict, decDictR, args): #1文ずつ処理していく　beamの時にbatch処理になるため、
def demo(model, encSents, decSents, encDictR, decDict, decDictR, args):
    for encSent, decSent in zip(encSents, decSents):
        print("Ques", " ".join([encDictR[z] if z != 0 else "<unk>" for z in encSent.tolist()]))
        print("gold", "".join([decDictR[z] if z != 0 else "<unk>" for z in decSent.tolist()]))
        beam = BeamDecording(model, encSent, decDict, args)
        for i, b in enumerate(beam):
            print("pred", i, b[0], "".join([decDictR[z] if z != 0 else "<unk>" for z in b[1]]))
        print("----------------------------------------------------")
            
            
            
    
if __name__ == "__main__":
    """main program"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu',
        dest='gpu',
        default=0,
        type=int,
        help='GPU ID')
    parser.add_argument(
        '-T',
        '--train-test-mode',
        dest='mode',
        default='train',
        help='select train or test')
    parser.add_argument(
        '-D',
        '--embed-dim',
        dest='embedDim',
        default=512,
        type=int,
        help='dimensions of embedding layers in both encoder/decoder [int] default=512')
    parser.add_argument(
        '-H',
        '--hidden-dim',
        dest='hiddenDim',
        default=512,
        type=int,
        help='dimensions of all hidden layers [int] default=512')
    parser.add_argument(
        '-E',
        '--epoch',
        dest='epoch',
        default=13,
        type=int,
        help='number of epoch [int] default=13')
    parser.add_argument(
        '-B',
        '--batch-size',
        dest='batch_size',
        default=4,
        type=int,
        help='mini batch size [int] default=4')
    parser.add_argument(
        '--enc-data',
        dest='encDataFile',
        default='./datasets/10k_train/train.en',
        help='filename for encoder training')
    parser.add_argument(
        '--dec-data-file',
        dest='decDataFile',
        default='./datasets/10k_train/train.ja',
        help='filename for decoder trainig')
    parser.add_argument(
        '--enc-devel-data-file',
        dest='encDevelDataFile',
        default='./datasets/dev_data/dev.en',
        help='filename for encoder development')
    parser.add_argument(
        '--dec-devel-data-file',
        dest='decDevelDataFile',
        default='./datasets/dev_data/dev.ja',
        help='filename for decoder development')
    parser.add_argument(
        '--dropout-rate',
        dest='dropout_rate',
        default=0.2,
        type=float,
        help='dropout rate default=0.3')
    parser.add_argument(
        '--attention',
        dest='attention',
        default=True,
        type=int,
        help='attention on/off')
    parser.add_argument(
        '--beam-size',
        dest='beam_size',
        default=1,
        type=int,
        help='beam size in beam search decoding default=1')
    parser.add_argument(
        '--decode-len',
        dest='decode_len',
        default=70,
        type=int,
        help='max length while decoding')
        
    print("start")
    args = parser.parse_args()
    if args.gpu >= 0:
        import cupy as xp
        cuda.check_cuda_available()
        cuda.get_device(args.gpu).use()
        print("gpu number", args.gpu)
    else:
        import numpy as xp
        args.gpu = -1
    xp.random.seed(0)
    random.seed(0)

    if args.mode == 'train':
        chainer.global_config.train = True
        chainer.global_config.enable_backprop = True
        chainer.global_config.use_cudnn = "always"
        chainer.global_config.type_check = True
        #train_model(args)
    elif args.mode == 'test':
        chainer.global_config.train = False
        chainer.global_config.enable_backprop = False
        chainer.global_config.use_cudnn = "always"
        chainer.global_config.type_check = True
        args.dropout_rate = 0.0
        #ttest_model(args)

    ###### train部分を書くけど以下は後でtrain_model(args)に全て移す TODO
    ppD = PrepareData(0)
    encDict = ppD.makeDict(args.encDataFile)
    decDict = ppD.makeDict(args.decDataFile)
    encSentList = ppD.makeSentenceList(args.encDataFile, encDict)
    decSentList = ppD.makeSentenceList(args.decDataFile, decDict)
    encDictR = {v: k for k, v in encDict.items()} #id2word 通常の逆をする辞書 デモとかで使うかも 訓練中はいらない
    decDictR = {v: k for k, v in decDict.items()}

    encSentListDev = ppD.makeSentenceList(args.encDevelDataFile, encDict) #<unk>は802 (default=300kでの話)
    decSentListDev = ppD.makeSentenceList(args.decDevelDataFile, decDict) #<unk>は548 (default=300kでの話)

    print("finish loading")
    
    trainIter = iterators.SerialIterator(ppD.makeInput(encSentList, decSentList), args.batch_size, repeat=True, shuffle=True)

    model = EncoderDecoder(encDict, decDict, args)
    model.to_gpu()
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    print("finish init")
    batch = trainIter.next()
    updater = EncoderDecoderUpdater(trainIter, optimizer, args)
    trainer = training.Trainer(updater, (args.epoch, "epoch"))
    trainer.extend(extensions.ProgressBar(update_interval=1))
    trainer.run()
    ##### ここから下はupdaterに書くことになるかもな TODO
    es = [xp.array(x[0], dtype=xp.int32) for x in batch]
    ds = [xp.array(x[1], dtype=xp.int32) for x in batch]

    encEmbed = model.getEmbeddings(es, args)
    hy, cy, ys = model.encNStepLSTM(hx=None, cx=None, xs=encEmbed)
    decEmbed = F.pad_sequence(model.getEmbeddings(ds, args)).transpose([1, 0, 2])
    
    loss, result = model.trainBatch(es, ds, args)

    print("Fin")
            
def train_model(args):
    return 0

##
# serializers.save_npz("locate", model)
# serializers.load_npz("locate", model)
