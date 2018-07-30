# EEGBCI 運動想起課題 データセット

https://www.physionet.org/pn4/eegmmidb/

- このデータセットは、これらのデータの作成に使用されたBCI2000計測システムの開発者によって作成され、PhysioNetに提供されました。システムの説明は次のとおりです。
  - Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. IEEE Transactions on Biomedical Engineering 51(6):1034-1043, 2004. [In 2008, this paper received the Best Paper Award from IEEE TBME.]
- この資料を参照する際には、この刊行物および www.bci2000.org を引用し、PhysioNetの標準引用文も含めてください。
  - Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215-e220 [Circulation Electronic Pages; http://circ.ahajournals.org/cgi/content/full/101/23/e215]; 2000 (June 13).

このデータセットは、以下に説明するように、109人のボランティアから得られた1500もの1分または2分のEEG記録からなる。

## 実験プロトコル
被験者は、BCI2000システム（http://www.bci2000.org）を用いて64チャンネルのEEGを記録しながら、異なる運動想起課題を行った。 各被験者は、14回の実験を行った：ベースライン1回につき2回（目を開いて1回、目を閉じて1回）、次の4つの各作業を行った．

1. ターゲットは画面の左側または右側に表示されます。被験者はターゲットが消えるまで、対応する拳を開閉します。その後、被験者はリラックスする。
2. ターゲットは画面の左側または右側に表示されます。被験者はターゲットが消えるまで、対応する拳を開閉することを想像する。その後、被験者はリラックスする。
3. ターゲットは画面の上部または下部に表示されます。被験者はターゲットが消えるまで、両拳（ターゲットが上にある場合）または両足（ターゲットが下にある場合）の両方を開閉します。その後、被験者はリラックスする。
4. ターゲットは画面の上部または下部に表示されます。被験者はターゲットが消えるまで、両拳（ターゲットが上にある場合）または両足（ターゲットが下にある場合）の開閉を想像します。その後、被験者はリラックスする。

要約すると、実験は以下の通りであった

1. Baseline, eyes open
1. Baseline, eyes closed
1. Task 1 (open and close left or right fist)
1. Task 2 (imagine opening and closing left or right fist)
1. Task 3 (open and close both fists or both feet)
1. Task 4 (imagine opening and closing both fists or both feet)
1. Task 1
1. Task 2
1. Task 3
1. Task 4
1. Task 1
1. Task 2
1. Task 3
1. Task 4

## モンタージュ
EEGは、以下に示すように、国際10-10システム（電極Nz、F9、F10、FT9、FT10、A1、A2、TP9、TP10、P9、およびP10を除く）に従って64個の電極から記録した 図）。 各電極名の下の数字は、それらがレコードに現れる順序を示す。 レコードの信号には0〜63の番号が付けられ、図の番号には1〜64の番号が付いています。

![EEG](https://www.physionet.org/pn4/eegmmidb/64_channel_sharbrough-old.png)
