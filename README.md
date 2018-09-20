# Transformer LM

TransformerのEncoderを使った言語モデル．  
Transformer自体の実装は以下．  
https://github.com/jadore801120/attention-is-all-you-need-pytorch

/data/ : ptbのデータ  
/src/ : ソースコード

## 中身
### trainer.py
学習，評価の管理

### lm.py
ロスと尤度を計算する言語モデル  
Models.pyに含まれるEncoderクラスを使っている．
Encoderの出力に単語ベクトルを内積とってsoftmax.  

### Models.py
上記リポジトリからの借用．  

単語埋め込み，Attentionをはる

i番目の単語を予測するときにi+1番目以降を見ないようにマスキングする部分を付け加えている．
マスキング部分はDecoderクラスから借用．  

#### パラメータ
n_src_vocab　語彙数
len_max_seq　最大文長
d_word_vec　単語ベクトルサイズ
n_layers　レイヤ数
n_heads　head数
d_k　
d_v　
d_model　出力サイズ=単語ベクトルサイズ
d_inner


#### 入力

idLines:padding済みの単語ID列  
0をpadding_idxとすると，  
```
[[8,9,2,4,6]  
 [9,2,0,0,0]
 [4,7,3,0,0]
```

inputLens:文の長さ
```
[[1,2,3,4,5]  
 [1,2,0,0,0]
 [1,2,3,0,0]
```

#### 出力
各単語位置に対応するd_modelサイズのベクトル

