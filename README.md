# 人工智能F组的作业代码库
## 小组成员
|姓名|学号|
|-----|-----|
|唐力|F38|
|裴哲论|F33|
|胡波|F07|
|郭立志|C11|
|王海滨|F40 |
|丁玮|F04 |
|熊彦勇|E46|
|巩振| |

## 第一次作业
### 第一题 动物识别系统

一、目的实现基于产生式规则的专家系统，需要体现专家系统的基本结构。  
二、用选定的编程语言构造规则库和综合数据库，并能对它们进行增加、删除和修改操作。  
三、简单动物识别系统总共能识别7种动物，即：老虎、金钱豹、斑马、长颈鹿、企鹅、鸵鸟、信天翁。建立识别七种动物识别系统的规则：  
由于实验要求系统的规则库和综合数据库能够进行增加、删除和修改操作，因此可以采取逐步添加条件，压缩范围的方法进行识别，即：  
先跟据一些动物的共性进行大致分类，然后在添加约束条件，将范围缩小，直到能够识别出每一种不同的动物为止。这样，我们在需要添加识别其他动物的功能时， 只需要添加那些动物的个性方面的信息即可，这样也可以简化系统的结构和内容。识别老虎、金钱豹、斑马、长颈鹿、企鹅、鸵鸟、信天翁这7种动物的简单动物识别系统规则一共可以写成以下15条：  
Rule1 IF 该动物有毛发 THEN 该动物是哺乳动物  
Rule2 IF 该动物有奶 THEN 该动物是哺乳动物  
Rule3 IF该动物有羽毛 THEN 该动物是鸟  
Rule4 IF 该动物会飞 AND会下蛋  THEN 该动物是鸟  
Rule5 IF 该动物吃肉 THEN 该动物是肉食动物  
Rule6 IF 该动物有犬齿AND有爪 AND眼盯前方THEN该动物是肉食动物  
Rule7 IF 该动物是哺乳动物 AND 有蹄 THEN 该动物是有蹄类动物  
Rule8 IF 该动物是哺乳动物 AND 是嚼反刍动物 THEN 该动物是有蹄类动物  
Rule9 IF 该动物是哺乳动物 AND 是肉食动物 AND 是黄褐色AND 身上有暗斑点THEN该动物是金钱豹  
Rule10 IF 该动物是哺乳动物 AND 是肉食动物 AND 是黄褐色AND 身上有黑色条纹 THEN 该动物是老虎  
Rule11 IF 该动物是有蹄类动物 AND 有长脖子 AND 有长腿 AND 身上有暗斑点 THEN 该动物是有长颈鹿  
Rule12 IF 该动物是有蹄类动物 AND 身上有黑色条纹 THEN 该动物是斑马  
Rule13 IF 该动物是鸟 AND 有长脖子 AND 有长腿 AND不会飞 THEN 该动物是鸵鸟  
Rule14 IF 该动物是鸟 AND 会游泳 AND 有黑白二色 AND不会飞 THEN 该动物是企鹅  
Rule15 IF 该动物是鸟 AND 善飞 THEN 该动物是信天翁  


### 第二题 野人牧师题

有n个牧师和n个野人准备渡河，但只有一条能容纳c个人的小船，为了防止野人侵犯牧师，要求无论在何处，牧师的人数不得少于野人的人数，除非牧师人数为0，且假定野人与牧师都会划船，试用程序语言设计一个算法，根据n和c的不同取值，确定他们能否渡过河去，若能，则给出小船来回次数最少的最佳方案，并用状态空间方法表示出来（记录过程）。

---

## 第二次作业：搜索策略

### 一  实验内容
1.	熟悉和掌握启发式搜索的定义、估价函数和算法过程；比较不同算法的性能。
2.	以重排九宫问题为例演示某种搜索策略的搜索过程，争取做到直观，清晰。  
重排九宫的问题定义：在一个3×3的方格棋盘上放置8个标有1、2、3、4、5、6、7、8数字的将牌，留下一个空格（用0表示）。规定与空上下左右相邻的将牌可以移入空格。问题要求寻找一条从某初始状态S0到目标状态Sg的将牌移动路线。  
改变其启发函数定义，观察结果的变化，分析原因。
3.  熟悉和掌握各种搜索策略的思想，掌握广度优先、深度优先、局部最优、全局最优、A*算法的定义、估价函数和算法过程，理解求解流程和搜索顺序。

### 二  实验思路
1．分别以深度优先、局部最优、全局最优、A*等搜索算法（任选两种算法）为例演示搜索过程，分析各种算法中的OPEN表CLOSE表的生成过程，分析估价函数对搜索算法的影响，分析某种启发式搜索算法的特点。进入演示系统后，选择搜索策略演示程序。实验步骤如下：  
(1) 选择不同的搜索算法， 观察搜索过程。  
(2) 设置不同属性，观察搜索过程的变化。  
(3)观察运行过程和搜索顺序，理解启发式搜索的原理。  
(4)算法流程的任一时刻的相关状态,以算法流程中各步骤open表、close表、节点静态图、效率等   
(5)对算法所选的两种算法进行对比分析。  
请大家按分组小组协作完成，组长组织小组内部分工协作，一定要协作，开发工具任选，本实验在23日讲授结束后分组演示。  