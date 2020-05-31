function varargout = gmm(train_images, K_or_centroids)
% ============================================================
 % Expectation-Maximization iteration implementation of
 % Gaussian Mixture Model.
 %
 % PX = GMM(X, K_OR_CENTROIDS)
 % [PX MODEL] = GMM(X, K_OR_CENTROIDS)
 %
 %  - X: N-by-D data matrix.----------NxD的矩阵
 %  - K_OR_CENTROIDS: either K indicating the number of--------单个数字K/[K] 或者 KxD矩阵的聚类
 %       components or a K-by-D matrix indicating the
 %       choosing of the initial K centroids.
 %
 %  - PX: N-by-K matrix indicating the probability of each--------NxK矩阵，第N个数据点占第K个单一高斯概率密度
 %       component generating each point.
 %  - MODEL: a structure containing the parameters for a GMM:
 %       MODEL.Miu: a K-by-D matrix.-------------KxD矩阵，初始化聚类样本，后面循环为每个数据点概率归一化后再聚类概率归一化后的均值矩阵
 %       MODEL.Sigma: a D-by-D-by-K matrix.------DxDxK矩阵，初始化数据点对于聚类的方差[聚类等概率]，后面循环为均值矩阵改变以后的方差
 %       MODEL.Pi: a 1-by-K vector.-----------1xK矩阵，初始化数据点使用聚类的概率分布,后面循环高斯混合概率系数归一化的分母Nk/N,高斯混合加权系数向量
 % ============================================================
 
     threshold = 1e-15;%阈值
     [N, D] = size(train_images);%矩阵X的行N，列D
  
     if isscalar(K_or_centroids)%判断值是否为1x1矩阵即单个数字
         K = K_or_centroids;%取[k]的k
         % randomly pick centroids
        rndp = randperm(N);%返回一行由1到N个的整数无序排列的向量
         centroids = train_images(rndp(1:K), :);%取出X矩阵行打乱后的矩阵的前K行
     else
        K = size(K_or_centroids, 1);%取矩阵K_or_centroids的行数
         centroids = K_or_centroids;%取矩阵K_or_centroids矩阵
     end
 
     % initial values
     [pMiu pPi pSigma] = init_params();
 %初始化 嵌套函数后面可见。KxD矩阵pMiu聚类采样点，1*K向量pPi使用同一个聚类采样点出现概率，D*D*K的矩阵pSigma矩阵X的列项j对于聚类采样点的协方差
  
     Lprev = -inf; %inf表示正无究大，-inf表示为负无究大
     while true
         Px = calc_prob();%NxK矩阵Px存放聚类点k（共有聚类点K个）对于全部数据点的正态分布生成样本的概率密度
  
         % new value for pGamma
        pGamma = Px .* repmat(pPi, N, 1);%NxK矩阵pGamma在使用聚类采样点k的条件下某个数据点n生成样本概率密度（条件概率密度）
         pGamma = pGamma ./ repmat(sum(pGamma, 2), 1, K); %NxK矩阵pGamma对于使用数据点条件概率密度行向归一化。
      %求每个样本由第K个聚类，也叫“component“生成的概率）
 
         % new value for parameters of each Component
         Nk = sum(pGamma, 1);%1xK矩阵Nk第k个聚类点被数据点使用的概率总和
         pMiu = diag(1./Nk) * pGamma' * train_images;
 %KxD矩阵 重新计算每个component的均值 维数变化KxK*KxN*NxD=KxD 数据点进行 聚类概率归一化.条件概率密度.数据点=得到均值（期望）
 %均值=Σ概率Pi*数据点某一属性Xi；而这里还多了个聚类概率Nk
        pPi = Nk/N; %更新混合高斯的加权系数
        for kk = 1:K %重新计算每个component的协方差矩阵
             Xshift = train_images-repmat(pMiu(kk, :), N, 1);%NxD矩阵Xshift在某一个聚类点的情况下，每个属性在这个聚类下的对均值（期望）差数(X-μi)
             pSigma(:, :, kk) = (Xshift' * ...
                 (diag(pGamma(:, kk)) * Xshift)) / Nk(kk);%DxD矩阵pSigma(::,i) 概率Pi  聚类概率=1/Nk(i)
                 %第i个方差矩阵= Σ(X-μi)转置*概率Pi*(X-μi)*第i个聚类概率
         end
  
         % check for convergence
         L = sum(log(Px*pPi')); %求混合高斯分布的似然函数
        if L-Lprev < threshold %随着迭代次数的增加，似然函数越来越大，直至不变
             break; %似然函数收敛则退出
         end
        Lprev = L;
     end
 
     if nargout == 1 %如果返回是一个参数的话，那么varargout=Px;
         varargout = {Px};
     else %否则，返回[Px model],其中model是结构体
        model = [];
         model.Miu = pMiu;
         model.Sigma = pSigma;
         model.Pi = pPi;
         varargout = {Px, model};
     end
 

     function [pMiu pPi pSigma] = init_params()
         pMiu = centroids;%得X矩阵中的任意K行，KxD矩阵  聚类点
         pPi = zeros(1, K);%获取K维零向量[0 0 ...0]     加权系数（每行数据与聚类中点最小距离的概率）
         pSigma = zeros(D, D, K);%获取K个DxD的零矩阵
  
         %distmat为D维距离差平方和
         % hard assign x to each centroids  %X有NxD；sum(X.*X, 2)为Nx1； repmat(sum(X.*X, 2), 1, K)行向整体1倍，列向整体K倍；结果NxK
        distmat = repmat(sum(train_images.*train_images, 2), 1, K) + ... %distmat第j行的第i个元素表示第j个数据与第i个聚类点的距离，如果数据有4个，聚类2个，那么distmat就是4*2矩阵
            repmat(sum(pMiu.*pMiu, 2)', N, 1) - 2*train_images*pMiu'; %sum(A，2)结果为每行求和列向量，第i个元素是第i行的求和；
        [dummy labels] = min(distmat, [], 2); %返回列向量dummy和labels，dummy向量记录distmat的每行的最小值，labels向量记录每行最小值的列号（多个取第一个），即是第几个聚类，labels是N×1列向量，N为样本数
 
         for k=1:K
             Xk = train_images(labels == k, :); %把标志为同一个聚类的样本组合起来
            pPi(k) = size(Xk, 1)/N; %求混合高斯模型的加权系数，pPi为1*K的向量  
            pSigma(:, :, k) = cov(Xk); 
		%如果X为Nx1数组，那么cov(Xk)求单个高斯模型的协方差矩阵
		%如果X为NxD(D>1)的矩阵，那么cov(Xk)求聚类样本的协方差矩阵
		%cov()求出的为方阵--《概率论与数理统计》-多维随机变量的数字特征，且是对称矩阵（上三角和下三角对称）--《线性代数》
		%pSigma为D*D*K的矩阵
        end
     end
  
     function Px = calc_prob()
         Px = zeros(N, K);%NxK零矩阵
         for k = 1:K
             Xshift = train_images-repmat(pMiu(k, :), N, 1); %NxD矩阵Xshift表示为对于一个1xD聚类点向量行向增N倍的样本矩阵-Uk，第i行表示xi-uk
            inv_pSigma = inv(pSigma(:, :, k)); %求协方差矩阵的逆，pSigmaD*D*K的矩阵， inv_pSigmaD*D的矩阵,Σ^-1
             tmp = sum((Xshift*inv_pSigma) .* Xshift, 2); 
     	%tmp为N*1矩阵，第i行表示（xi-uk)^T*Sigma^-1*(xi-uk) 即-(x-μ)转置x(Σ^-1).(x-μ)   ----矩阵有叉乘（矩阵乘）和点乘
            coef = (2*pi)^(-D/2) * sqrt(det(inv_pSigma));
		 %求多维正态分布中指数前面的系数，（2π)^(-d/2)*|(Σ^-(1/2))|
            Px(:, k) = coef * exp(-0.5*tmp); %NxK矩阵求单独一个D维正态分布生成样本的概率密度或贡献
        end
     end
 end
%聚类  数据挖掘
%矩阵算法 线性代数
%概率密度，近似然  数理统计