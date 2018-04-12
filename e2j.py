import sys
import collections
import six.moves.cPickle as pickle
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
        ys = F.pad_sequence(ys) #[batch, max(sentlen), Dim]

        ###decoder step
        decEmbed = self.getEmbeddings(decSents, args) #embed のリストの状態
        decEmbed = F.pad_sequence(decEmbed).transpose([1, 0, 2]) #padding して[sentLen, batch, Dim]に変更

        decod_step = len(decEmbed) - 1
        decoderOutList = [0] * decod_step
        lstmStateList = [0] * decod_step
        firstInput = chainer.Variable(xp.zeros(hy[0].shape, dtype=xp.float32)) #decLSTMの最初に入力として初期化 embedじゃない方
        for i in range(decod_step):
            if i == 0: #デコーダの最初のステップ
                self.set_state([cy[0], hy[0]])
                anoInput = firstInput
            else:
                self.set_state(lstmStateList[i - 1])
                anoInput = decoderOutList[i - 1]
            hOut = self.decLSTM(F.concat([decEmbed[i], anoInput], 1)) #decoder LSTMの出力
            lstmStateList[i] = self.get_state()
            decoderOutList[i] = attention(hOut, ys, args) #decoder LSTMの出力をアテンションしたもの decoderの出力 decod_step * [batch, Dim]

        ###output層
        correctLabels = F.pad_sequence(decSents, padding=-1).T.array #TODO 何か嫌だから上手く書きたい　ってかこの関数全体何か汚い -1でパディングしたから1足したら0がeosトークンになるんじゃね？
        for i in range(decod_step):
            oVector = self.decOut(F.dropout(decoderOutList[i], args.dropout_rate))
            correctLabel = correctLabels[i + 1]

            proc += (xp.count_nonzero(correctLabel + 1)) ###TODO 0を数えてたらunkトークンがなくなるし、1足したら全部1以上になるンゴ
            # 必ずminibatchsizeでわる
            closs = chaFunc.softmax_cross_entropy(
                oVector, correctLabel, normalize=False) #normalize=Falseの意味？ paddingしてるからっぽい
            # これで正規化なしのloss  cf. seq2seq-attn code
            total_loss_val += closs.data * cMBSize
            if train_mode > 0:  # 学習データのみ backward する
                total_loss += closs
            # 実際の正解数を獲得したい
            t_correct = 0
            t_incorrect = 0
            # Devのときは必ず評価，学習データのときはオプションに従って評価
            if train_mode == 0 or args.doEvalAcc > 0:
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
        if train_mode > 0:  # 学習時のみ backward する 学習するのはvalじゃない方　valは.dataだから当たり前か
            total_loss.backward()

        return total_loss_val, (correct, incorrect, decoder_proc, proc)
            
    def attention(self, hOut, ys, args): #TODO この関数も何か汚い
        """calc attention"""
        if args.attention == 0: #アテンションをしない時hOutをそのまま返す
            return hOut
        #ターゲット側の下準備
        hOut1 = self.attnIn(hOut) #[batch, Dim]
        hOut2 = F.expand_dims(hOut1, axis=1) #[batch, 1, Dim]
        hOut3 = F.broadcast_to(hOut2, (len(hOut2), len(ys[0]), args.hiddenDim)) #[batch, max(enc_sentlen), Dim] 今ysとhOut3は同じshapeのはず

        aval = F.sum(ys*hOut3, axis=2) #[batch, sentlen]

        cAttn1 = F.softmax(aval) #[batch, max(enc_sentlen)] paddingで0のところはかなり小さい数字の確率で出てくる
        cAttn2 = F.expand_dims(cAttn1, axis=1) #[batch, 1, max(enc_sentlen)]
        cAttn3 = F.batch_matmul(cAttn2, ys) #[batch, 1, Dim]
        context = F.reshape(cAttn3, (len(ys), len(ys[0][0]))) #[batch, Dim] エンコーダコンテキストベクトルの完成

        c1 = F.concat((hOut, context)) #[batch, Dim + Dim]
        c2 = self.attnOut(c2) #[bathc, Dim]
        return F.tanh(c2) #活性化
        
    
    def getEmbeddings(self, sentenceList, args):
        sentenceLen = [len(sentence) for sentence in sentenceList]
        sentenceSection = xp.cumsum(sentenceLen[:-1])
        sentenceEmbed = F.dropout(self.encEmbed(F.concat(sentenceList, axis=0)), args.dropout_rate)
        return F.split_axis(sentenceEmbed, sentenceSection, 0, force_tuple=True)

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

    
if __name__ == "__main__":
    """main program"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu',
        dest='gpu',
        default=1,
        type=int,
        help='GPU ID for model')
    parser.add_argument(
        '-T',
        '--train-test-mode',
        dest='train_mode',
        default='train',
        help='select train or test')
    parser.add_argument(
        '-D',
        '--embed-dim',
        dest='embedDim',
        default=512,
        type=int,
        help='dimensions of embedding layers in both encoder/decoder ')
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
        default='./datasets/300k_train/train.en',
        help='filename for encoder training')
    parser.add_argument(
        '--dec-data-file',
        dest='decDataFile',
        default='./datasets/300k_train/train.ja',
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
        
    print("start")
    args = parser.parse_args()
    # if args.gpu >= 0:
    #     import cupy as xp
    #     cuda.check_cuda_available()
    #     cuda.get_device(args.gpu).use()
    #     print(args.gpu)
    # else:
    #     import numpy as xp
    #     args.gpu = -1
    ###変更せよ TODO
    xp = np
    xp.random.seed(0)
    random.seed(0)

    if args.train_mode == 'train':
        chainer.global_config.train = True
        chainer.global_config.enable_backprop = True
        chainer.global_config.use_cudnn = "always"
        chainer.global_config.type_check = True
        #train_model(args)
    elif args.train_mode == 'test':
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
    print("finish init")
    ##### ここから下はupdaterに書くことになるかもな TODO
    batch = trainIter.next()
    es = [xp.array(x[0], dtype=xp.int32) for x in batch]
    ds = [xp.array(x[1], dtype=xp.int32) for x in batch]

    encEmbed = model.getEmbeddings(es, args)
    hy, cy, ys = model.encNStepLSTM(hx=None, cx=None, xs=encEmbed)
    decEmbed = F.pad_sequence(model.getEmbeddings(ds, args)).transpose([1, 0, 2])



    print("Fin")
            
def train_model(args):
    return 0

