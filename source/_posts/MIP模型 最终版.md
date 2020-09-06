---
title: 电动汽车充电最优化的MIP模型
---



## 电动汽车全日充电最优化的MIP建模



## 介绍

混合整数规划是变量中既含有连续变量也含有{0,1}整数变量的优化问题。这类问题是NP-hard问题，算法复杂度随着变量增加呈指数级增长。混合整数规划MIP的建模由三个要素组成：**决策变量，约束条件，目标函数**。数学上的表现形式如下

![image-20200827153256004](C:\Users\GTBA1C1N\AppData\Roaming\Typora\typora-user-images\image-20200827153256004.png)

MIP问题的求解对模型的表达极其敏感，建模的复杂度取决了求解器能否找到解，以及求解所需的时间。一旦模型成功建立好后，将由整数规划求解器如(Cplex, Gurobi, SICP) 等使用**分支界定法**(branch and bound) 算法求解。分支界定法的本质是在树上对所有可能的状态**搜索**, 以及用上界和下界**剪枝**去掉不符合约束限制和目标函数值大于当前找到的最小值的状态起到加速作用。

![image-20200827152509808](C:\Users\GTBA1C1N\AppData\Roaming\Typora\typora-user-images\image-20200827152509808.png)

在该建模中，最终的优化目标是找到一个充电方案，使所有输入的汽车能够：

1. 满足汽车的运营需求

2. 充电成本最低

为此，模型的思路是：**限制充电的电量，使巴士在离站时的电量严格> 30%**。因此如果巴士在一天的运营里电量都大于30%该巴士则不需充电；如果巴士电量有掉到30%以下的可能，程序会选择最便宜的时间段提前充最少的点使电量刚刚好大于30%。该模型对30辆巴士，24小时长的时间段能在10分钟内找到离最优解差3%以内的解。针对复杂情况85辆巴士，32小时长的时间段能在1小时50分钟找到离最优解差5%的可行解。

刘海东针对深圳巴士的建模是针对深圳巴士特定场景的，解决了深圳巴士从21点起夜间进站至次日7点离站10小时的**单个时间段**内，所有巴士如何将电池充满。而本建模在上述建模的基础上做出了改进，**对变量进行了扩充并添加了约束条件**，从而扩展到了32小时内巴士能多次进出的情况，在**多个时间段**里使得电池在每次出站时维持安全的电量SOC > 30%。

## 变量 

### **时间相关变量**

- T 记录从一日开始00:00 到次日早上08:00的L个时间段，一个时间段的粒度为5min。粒度的选择得在精度和求解速度上做权衡，5分钟的粒度是符合应用精度和计算速度要求的选择。

  $$
  \delta = 5min \\
  T = [0\ 1 ... L-1]
  $$
  
- Z 每个时间段对应的电费

   $$
  Z = [z_0 \ z_1 ... z_{L-1}]  \ z_i\in \{0.23,0.67,1.02\}
  $$
  

### 充电桩变量

* 需要注意的是，值一样的对称变量会创造很多重复的状态，减速求解过程。比如，1号桩和2号桩的功率假设一样，巴士选择在1号充或是在2号充的目标函数值是一样的，但却产生了两种解。而在深圳巴士的充电桩中存在着大量充电功率一样的充电桩，因此为了减少对称解，我们将充电功率一样的桩聚合成一类，并将这一类下的桩记录下来

  ```python
  dict_pile = {power_1: [pileNum1,pileNum2],  .... ,power_n:[pileNumN]}
  ```

  

- C 总共有N类充电桩，每个充电桩的输出功率。单位为kW。充电过程假设为线性。

  
  $$
  C = [power_1 \ power_2 ... power_{N-1}]_N
  $$
  

### 巴士变量

- X: 巴士的进站时间段，对应输入excel文件里的T_Arrival

  Y: 巴士的出战时间段，对应输入excel文件里的T_Departure

  PRP: Possible recharging period 表示巴士在站可以充电的时间段

  ```
  X = {bus1:[x10, x11,...,x1k], ..., busM:[xM0...xMj]}
  Y = {bus1:[y10, y11,...,y1k], ..., busM:[yM0...yMj]}
  PRP = {bus1:[[x10,y10], ..., [x1k,y1k]],..., busM:[[xM0,yM0],...,[xMj,yMj]]}
  ```

- 巴士电池容量及每公里耗电，目前简单地取了深圳巴士的数据，容量为常数260kWh，每公里耗电为1.1kWh/km。后续可能为其他巴士提供更复杂的输入。

- Conso: 对应了excel输入文件里distance_run的消耗电量。为方便起见，只在进站的时间段计算上一段运营距离的耗电量，不在运营时计算实时的耗电量（不在站时的耗电量为0）。

  ```
  Conso = {bus1:[0 0 0 1.1*Distance_Run[0] ... 1.1] ... }
  ```

- alpha: 每辆汽车可以在站充电的时间段数目

   ```
   alpha = {bus1: len(PRP[bus1]), ..., busM:len(PRP[busM])}
   ```

   

### 决策变量

- B：表示巴士什么时候开始充电的变量。 巴士m总共有alpha[m]个可充电的时间段，m至多可以选择在一个可充时间段里充电1次，在这个可充时间段内m可以选择时间点t开始充电。这样保证了一辆车在一个可充时间段内可以固定连续的在1个充电桩上充电。

  $$
  B = [B_{tml}]_{alpha[m]*M*L} \\
  b_{tml} = 1  \    第m辆巴士连在它第l个在站时间段的第t个时间点上开始充电
  $$
  
```python
   B = [[[bcMdl.add_var(name="b,{},{},{}".format(t, m, l), var_type=BINARY) for l in range(alpha[m])] for m in range(M)] for t in range(L)]
```

- D: 表示巴士什么时候结束充电的变量。与B对应，一次充电开始必然对应一次充电结束。如果巴士m选择在一个可充时间段内充电，在这个可充时间段内必须有一个时间点巴士停止充电。
  $$
  D = [d_{tml}]_{alpha[m]*M*L} \\
  d_{tml} = 1  \    第m辆巴士连在它第l个在站时间段的第t个时间点上结束充电
  $$

  ```python
   D = [[[bcMdl.add_var(name="d,{},{},{}".format(t, m, l), var_type=BINARY) for l in range(alpha[m])] for m in range(M)] for t in range(L)]
  
  ```

- P： 表示巴士和充电桩在某所有时间点上的连接状态。可以获得巴士和充电桩的匹配信息。重要的结果输出。
  $$
  P = [p_{ijk}]_{M*N*L} \\
  p_{ijk} = 1 在第k个时间段，第i辆巴士连在第j个充电桩上充电
  $$

  ```python
      P = [[[bcMdl.add_var(name="p,{},{},{}".format(t, n, m), var_type=BINARY)
             for m in range(M)] for n in range(N)] for t in range(L)]
  ```

- Q： 表示巴士在一个可充时间段内和充电桩的匹配状态。主要用于约束里限制巴士在一个小可充时间段里只能跟一个充电桩匹配，进而减少了对称解。
  $$
  Q = [p_{ijk}]_{M*N*alpha[m]} \\
  q_{ijk} = 1 第i辆巴士在它的第k个在站时间段连在第n个充电桩上充电
  $$
  

  ```python
  Q = [[[bcMdl.add_var(name="q,{},{},{}".format(n, m, l), var_type=BINARY) for l in range(alpha[m])] for m in range(M)] for n in range(N)]
  ```

- SOC：表示每辆车在每个时间段上的电量。重要的结果输出。

  ```python
   SoC = [[bcMdl.add_var(name="SoC,{},{}".format(t, m), var_type=CONTINUOUS) for m in range(M)] for t in range(L)]
  ```

  



## 约束条件

1. **巴士在一个在站时间段最多只能充电一次** ：至多一次开始，一次结束
   $$
   \sum_{t}B_{tml} \leq 1  \  \ \forall m, l \\
   \sum_{t}D_{tml} \leq 1  \  \ \forall m, l \\
$$
   
```python
   # 3.1 Each bus has at most one charge beginning time frame for each period
   for m in range(M):
       for l in range(alpha[m]):
           bcMdl += xsum(B[t][m][l] for t in range(L)) <= 1
   # 3.2 Each bus has one and only one charge end time frame
   for m in range(M):
       for l in range(alpha[m]):
           bcMdl += xsum(D[t][m][l] for t in range(L)) <= 1
   
   ```
   

   
2. **巴士在一个在站时间段里只能选择跟一个充电桩匹配充电** 

   如果在这个时间段Q对所有桩的和是0，巴士在这个时间段不充电，在这个时间段内P的和也应为0.
   
   否则，Q对所有桩的和是1，巴士跟一个桩充电，P在这个时间段内的和应不超过5分钟粒度的个数
   $$
   \sum_{n}Q_{mnl} \leq 1  \  \forall m,l \\
   \sum_{t \in PRP[m][l]} P_{tmn} \leq Q_{nml}\times len(PRP[m][l])   \    \forall m,n,l
   $$
   

```python
    # 3.3 Each bus uses one and only one pile within the optimization target time period
    for m in range(M):
        for l in range(alpha[m]):
            bcMdl += xsum(Q[n][m][l] for n in range(N)) <= 1
            
    for (n, m) in product(range(N), range(M)):
        for l in range(len(PRP[m])):
            bcMdl += xsum(P[t][n][m] for t in PRP[m][l]) <= len(PRP[m][l]) * Q[n][m][l]

```



3. **每类桩在一个时间点上不能充超出桩数目的车** 
   $$
   \sum_{m}P_{tnm} \leq Nb_{pileCluster[n]}  \  \forall t,n \\
   \sum_{n}P_{tnm} \leq Nb_{pileCluster[n]}  \  \forall t,m \\
   $$
   

   ```python
       # 3.4 Each pile cluster charges at most buses with the maximum nb of piles of the same power.
       for (t, n) in product(range(L), range(N)):
           bcMdl += xsum(P[t][n][m] for m in range(M)) <= len(dict_pile[C[n]])
   
       for (t, m) in product(range(L), range(M)):
           bcMdl += xsum(P[t][n][m] for n in range(N)) <= len(dict_pile[C[n]])
   ```

4. **巴士开始充电的时间应晚于进站时间，结束充电的时间应早于出站时间且晚于开始充电的时间**

   通过t乘以B或D可以得到巴士开始/结束充电的时间点
   $$
   \sum_{t \in PRP[m][l]}tB_{tml} \geq X[m][l]  \ \forall m,l \\
   \sum_{t \in PRP[m][l]}tD_{tml} \leq Y[m][l]  \ \forall m,l \\
   \sum_{t \in PRP[m][l]}tD_{tml} \geq \sum_{t \in PRP[m][l]}tB_{tml}  \ \forall m,l \\
   $$
   

   ```python
       # 3.5 Charge start time is later than enter station time for each bus
       for m in range(M):
           for l in range(len(PRP[m])):
               bcMdl += xsum(t * B[t][m][l] for t in PRP[m][l]) >= int(X[m][l])
   
       # 3.6 Charge stop time is not earlier than charge start time for each bus
       for m in range(M):
           for l in range(len(PRP[m])):
               bcMdl += xsum(t * D[t][m][l] for t in PRP[m][l]) >= xsum(t * B[t][m][l] for t in PRP[m][l])
   
       # 3.7 Exit station time is later than charge stop time for each bus
       for m in range(M):
           for l in range(len(PRP[m])):
               bcMdl += xsum(t * D[t][m][l] for t in PRP[m][l]) <= int(Y[m][l])
   
   ```

5. **SOC应在[0%,100%]的区间内，且每次出站时SOC应大于30%。 假设所有巴士起始的SOC时是100%**

   ```python
       ## 3.8 Charged energy is not less than energy demand for each bus within the optimization target time period
       for m in range(M):
           bcMdl += SoC[0][m] == 260 #起始SOC假设为100%
           for t in range(L):
               bcMdl += SoC[t][m] >= 0 # >0%
           	if t!= 0:
               	bcMdl += SoC[t][m] == SoC[t-1][m] - Conso[m][t] + delta * xsum(C[n] * P[t][n][m] for n in range(N)) #t时刻的SOC由t-1时刻SOC减去耗电并加上充电得到
               	bcMdl += SoC[t][m] <= 260 # < 100%
                   
               
       for m in range(M):
           for t_exit in Y[m]:
               bcMdl += SoC[t_exit][m] >= 0.3*260 #出站时SOC >30%
   ```

6. **P与其他变量的关系：P在[B,D]从开始充电到结束充电的区间里都为1， P在不在站的时间段里为0**
   $$
   a. \  \sum_{n} P_{tmn} - P_{(t-1)mn} \leq B_{tm} \\
   b. \ \frac{(\sum_{n} P_{tmn} - P_{(t-1)mn}) + 1 }{2}\geq B_{tm} \\
    当B_{tm}= 1时，我们需要P_{tmn} = 1, P_{(t-1)mn} = 0.\\
    为此我们取a和b的交集，当B_{tm} = 1时， 1\leq\sum_{n} P_{tmn} - P_{(t-1)mn}\leq1\\
    同理，当D_{tm}= 1时，我们需要P_{tmn} = 1, P_{(t+1)mn} = 0. 我们有类似的不等式取交集\\
    \sum_{n} P_{tmn} - P_{(t+1)mn} \leq D_{tm} \\
   \frac{(\sum_{n} P_{tmn} - P_{(t+1)mn}) + 1 }{2}\geq D_{tm} \\
   $$
   

   ```python
     # 3.10 Relationship between charge end status and charge status of each time frame for each bus
       
       for m in range(M):
           for l in range(len(X[m])):
               bcMdl += xsum(P[X[m][l]][n][m] for n in range(N)) == B[X[m][l]][m][l]
         
       for m in range(M):
           for l in range(len(X[m])):
               for t in range(X[m][l]+1,Y[m][l]):
                   bcMdl += xsum(P[t][n][m] - P[t-1][n][m] for n in range(N)) <= B[t][m][l]
                   bcMdl += xsum(P[t][n][m] - P[t-1][n][m] for n in range(N)) + 1 >= 2 * B[t][m][l]
                   
       for m in range(M):
           for l in range(len(Y[m])):
               bcMdl += xsum(P[Y[m][l]][n][m] for n in range(N)) == D[Y[m][l]][m][l]
         
       for m in range(M):
           for l in range(len(X[m])):
               for t in range(X[m][l], Y[m][l]-1):
                   bcMdl += xsum(P[t][n][m] - P[t+1][n][m] for n in range(N)) <= D[t][m][l]
                   bcMdl += xsum(P[t][n][m] - P[t+1][n][m] for n in range(N)) + 1 >= 2 * D[t][m][l]
                   
       for t in range(L):
           for m in range(M):
               Chargable = False
               for l in range(len(PRP[m])):
                   if t in PRP[m][l]:
                       Chargable = True
                       break
                       if not Chargable:
                           bcMdl += xsum(P[t][n][m] for n in range(N)) == 0 #不在站时间段P=0
   ```

   

## 目标函数

- 充电电费：电费 * 充电时间 * 充电功率 * 充电状态(1或0)

  $$
  min(\delta \sum_{m} \sum_{n} \sum_{t} c_n P_{tmn}z_t)
  $$
  
  
```python
  bcMdl.objective = minimize(delta * xsum(Z[t] * C[n] * P[t][n][m]
                             for t in range(L) for n in range(N) for m in range(M)))
```



## 结果

程序在多种情况的输入下进行了测试: 

* 从深圳巴士运营数据中抽取的从08：00到晚22：00的终点站在香梅北总站的18辆38路巴士，它们在停留在总站的短暂时间内可以充电。**找到可行解返回的时间约1分钟内**

* 从深圳巴士充电数据抽取的从22：00到翌日07：00在香梅北充电的61辆巴士。它们仅进站一次停留很长的时间，希望能将它们的电充满。**在输入需要加入额外的一行7：50-8:00 行驶距离163km的一行**（刚好消耗约电池70%的电量，使巴士在22:00 到7:00间能把电充满）。人工添加输入数据是短暂的解决方法，后续可能更改模型区分两类车减少工作。**找到可行解返回的时间约20分钟**

  ![image-20200828120241098](C:\Users\GTBA1C1N\AppData\Roaming\Typora\typora-user-images\image-20200828120241098.png)

* 复杂数据：前述的18辆日间巴士+61辆夜间巴士+ 6辆在夜间运营的人造数据巴士 共85辆巴士，从00：00至翌日08：00共32小时。 **找到可行解返回的时间约1分钟50分钟**



**输出数据：**以下是85辆车复杂情况的输出结果节选

* SOC：每辆巴士在每个时间段含有的电量，单位为kWh。

  ![image-20200828121220206](C:\Users\GTBA1C1N\AppData\Roaming\Typora\typora-user-images\image-20200828121220206.png)

* ChargeStatus：每辆巴士在每个时间段与选择与哪个功率的充电桩充电。

  <img src="C:\Users\GTBA1C1N\AppData\Roaming\Typora\typora-user-images\image-20200828121426832.png" alt="image-20200828121426832" style="zoom:67%;" />

  

* ChargePlan：每辆车的充电起始时间，充电电量，和连接的充电桩功率

  <img src="C:\Users\GTBA1C1N\AppData\Roaming\Typora\typora-user-images\image-20200828121539178.png" alt="image-20200828121539178" style="zoom:67%;" />

  <img src="C:\Users\GTBA1C1N\AppData\Roaming\Typora\typora-user-images\image-20200828121600505.png" alt="image-20200828121600505" style="zoom:67%;" />

## 下一步

* 本地图形化界面或部署到云端/服务器上
* 商业求解器Cplex和Gurobi的license如何解决。Gurobi暂时可以用学术版。后续可以尝试使用开源但效果稍差的SICP

- 分析并比较解，尤其是解的用电成本。使用charge_fee_cal.py即可导出结果
- 针对夜晚只充一次的巴士还需要手工在数据里添加一行，后续可以改模型简化此步
- 为了加速qiujiegqiujieguuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu输出的P给的对应的是1个功率的充电桩，

