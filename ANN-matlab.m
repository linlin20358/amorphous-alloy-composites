%% 初始化 
clear 
close all 
clc 
%% 读取读取 
ANN=load('ANNgai.txt');
ANNlabel=load('ANNAM.txt'); %改种类
N=length(ANNlabel); %全部样本数目
temp= randperm(size(ANN,1));
input_train = ANN(temp(1:0.8*N),[1,2,4])'; 
output_train =ANNlabel(temp(1:0.8*N))'; 
input_test=ANN(temp(0.8*N+1:end),[1,2,4])';
output_test=ANNlabel(temp(0.8*N+1:end))';
%% 数据归一化（输入、输出） 
[inputn,inputps]=mapminmax(input_train,0,1); 
[outputn,outputps]=mapminmax(output_train,0,1); 
inputn_test=mapminmax('apply',input_test,inputps);
inputn_train=mapminmax('apply',input_train,inputps);
%% 获取输入层节点、输出层节点个数 
inputnum=3; 
outputnum=1;
disp('/////////////////////////////////') 
disp('神经网络结构...') 
disp(['输入层的节点数为：',num2str(inputnum)])
disp(['输出层的节点数为：',num2str(outputnum)])
disp(' ') 
disp('隐含层节点的确定过程...') 
%确定隐含层节点个数 
%采用经验公式hiddennum=sqrt(m+n)+a，m为输入层节点个数，n为输出层节点个数，a一般取为1-10之间的整数 
MSE=1e+5; %初始化最小误差 
transform_func={'tansig','purelin'}; %激活函数 
train_func='trainlm'; %训练算法 
for hiddennum=fix(sqrt(inputnum+outputnum))+1:fix(sqrt(inputnum+outputnum))+10 
    %构建网络 
    net=newff(inputn,outputn,hiddennum,transform_func,train_func); 
    % 网络参数 
    net.trainParam.epochs=1000; % 训练次数 
    net.trainParam.lr=0.01; % 学习速率 
    net.trainParam.goal=0.000001; % 训练目标最小误差 
    % 网络训练 
    net=train(net,inputn,outputn); 
    an0=sim(net,inputn); %仿真结果 
    mse0=mse(outputn,an0); %仿真的均方误差 
    disp(['隐含层节点数为',num2str(hiddennum),'时，训练集的均方误差为：',num2str(mse0)]) 
    %更新最佳的隐含层节点 
    if mse0<MSE %%保证了不会“过拟合” 
        MSE=mse0; 
        hiddennum_best=hiddennum; 
    end
end
disp(['最佳的隐含层节点数为：',num2str(hiddennum_best),'，相应的均方误差为：',num2str(MSE)])
%% 构建最佳隐含层节点的BP神经网络 
net=newff(inputn,outputn,hiddennum_best,transform_func,train_func); 
%网络参数 
W1= net.iw{1,1};%输入层到中间层的权值
B1 = net.b{1};%中间各层神经元阈值 
W2 = net.lw{2,1};%中间层到输出层的权值
B2 = net.b{2};%输出层各神经元阈值
W3=W2*W1;
net.trainParam.epochs=1000;
net.trainParam.lr=0.01; % 学习速率 
net.trainParam.goal=0.000001; % 训练目标最小误差 
%% 网络训练 
net=train(net,inputn,outputn); %% 网络测试 
an=sim(net,inputn_test); %用训练好的模型进行仿真 
bn=sim(net,inputn_train);
test_simu=mapminmax('reverse',an,outputps); % 预测结果反归一化 
train_simu=mapminmax('reverse',bn,outputps);  
error=test_simu-output_test; %预测值和真实值的误差
%rsquare=Rsquare-cal(test_simu-output_test);
%%以上用于仿真，实际用于预测的时候只需将 *测试集* 对应的输入参数正常替换成待预测的输入参数，即可得出预测结果** 
%%真实值与预测值误差比较 
figure 
plot(output_test,'bo-','linewidth',1.2) 
hold on 
plot(test_simu,'r*-','linewidth',1.2)
hold on
plot(error,'square','MarkerFaceColor','b')
legend('期望值','预测值','误差') 
xlabel('测试样本编号'),ylabel('指标值') 
title('测试集预测值,期望值和误差') 
set(gca,'fontsize',12) 
figure 
plot(error,'ro-','linewidth',1.2) 
xlabel('测试样本编号'),ylabel('预测偏差') 
title('测试集的预测误差') 
set(gca,'fontsize',12) %计算误差 
[~,len]=size(output_test); 
SSE1=sum(error.^2); 
MAE1=sum(abs(error))/len; 
MSE1=error*error'/len; 
RMSE1=MSE1^(1/2); 
r=corrcoef(output_test,test_simu); %corrcoef计算相关系数矩阵，包括自相关和互相关系数
R1=r(1,2); 
h=R1^2;
%t=abs(1-test_simu);
%s=abs(1-train_simu);
%q=abs(1-valid_simu);
disp(' ') 
disp('/////////////////////////////////') 
disp('预测误差分析...') 
disp(['误差平方和SSE为： ',num2str(SSE1)]) 
disp(['平均绝对误差MAE为： ',num2str(MAE1)]) 
disp(['均方误差MSE为： ',num2str(MSE1)]) 
disp(['均方根误差RMSE为： ',num2str(RMSE1)]) 
disp(['相关系数R为： ',num2str(R1)]) 
%结果 
disp(' ') 
disp('/////////////////////////////////') 
disp('测试集预测结果') 
disp('编号 实际值 预测值 误差')
for i=1:len 
    disp([i,output_test(i),test_simu(i),error(i)]) 
end