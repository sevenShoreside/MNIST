function varargout = gmm(train_images, K_or_centroids)
% ============================================================
 % Expectation-Maximization iteration implementation of
 % Gaussian Mixture Model.
 %
 % PX = GMM(X, K_OR_CENTROIDS)
 % [PX MODEL] = GMM(X, K_OR_CENTROIDS)
 %
 %  - X: N-by-D data matrix.----------NxD�ľ���
 %  - K_OR_CENTROIDS: either K indicating the number of--------��������K/[K] ���� KxD����ľ���
 %       components or a K-by-D matrix indicating the
 %       choosing of the initial K centroids.
 %
 %  - PX: N-by-K matrix indicating the probability of each--------NxK���󣬵�N�����ݵ�ռ��K����һ��˹�����ܶ�
 %       component generating each point.
 %  - MODEL: a structure containing the parameters for a GMM:
 %       MODEL.Miu: a K-by-D matrix.-------------KxD���󣬳�ʼ����������������ѭ��Ϊÿ�����ݵ���ʹ�һ�����پ�����ʹ�һ����ľ�ֵ����
 %       MODEL.Sigma: a D-by-D-by-K matrix.------DxDxK���󣬳�ʼ�����ݵ���ھ���ķ���[����ȸ���]������ѭ��Ϊ��ֵ����ı��Ժ�ķ���
 %       MODEL.Pi: a 1-by-K vector.-----------1xK���󣬳�ʼ�����ݵ�ʹ�þ���ĸ��ʷֲ�,����ѭ����˹��ϸ���ϵ����һ���ķ�ĸNk/N,��˹��ϼ�Ȩϵ������
 % ============================================================
 
     threshold = 1e-15;%��ֵ
     [N, D] = size(train_images);%����X����N����D
  
     if isscalar(K_or_centroids)%�ж�ֵ�Ƿ�Ϊ1x1���󼴵�������
         K = K_or_centroids;%ȡ[k]��k
         % randomly pick centroids
        rndp = randperm(N);%����һ����1��N���������������е�����
         centroids = train_images(rndp(1:K), :);%ȡ��X�����д��Һ�ľ����ǰK��
     else
        K = size(K_or_centroids, 1);%ȡ����K_or_centroids������
         centroids = K_or_centroids;%ȡ����K_or_centroids����
     end
 
     % initial values
     [pMiu pPi pSigma] = init_params();
 %��ʼ�� Ƕ�׺�������ɼ���KxD����pMiu��������㣬1*K����pPiʹ��ͬһ�������������ָ��ʣ�D*D*K�ľ���pSigma����X������j���ھ���������Э����
  
     Lprev = -inf; %inf��ʾ���޾���-inf��ʾΪ���޾���
     while true
         Px = calc_prob();%NxK����Px��ž����k�����о����K��������ȫ�����ݵ����̬�ֲ����������ĸ����ܶ�
  
         % new value for pGamma
        pGamma = Px .* repmat(pPi, N, 1);%NxK����pGamma��ʹ�þ��������k��������ĳ�����ݵ�n�������������ܶȣ����������ܶȣ�
         pGamma = pGamma ./ repmat(sum(pGamma, 2), 1, K); %NxK����pGamma����ʹ�����ݵ����������ܶ������һ����
      %��ÿ�������ɵ�K�����࣬Ҳ�С�component�����ɵĸ��ʣ�
 
         % new value for parameters of each Component
         Nk = sum(pGamma, 1);%1xK����Nk��k������㱻���ݵ�ʹ�õĸ����ܺ�
         pMiu = diag(1./Nk) * pGamma' * train_images;
 %KxD���� ���¼���ÿ��component�ľ�ֵ ά���仯KxK*KxN*NxD=KxD ���ݵ���� ������ʹ�һ��.���������ܶ�.���ݵ�=�õ���ֵ��������
 %��ֵ=������Pi*���ݵ�ĳһ����Xi�������ﻹ���˸��������Nk
        pPi = Nk/N; %���»�ϸ�˹�ļ�Ȩϵ��
        for kk = 1:K %���¼���ÿ��component��Э�������
             Xshift = train_images-repmat(pMiu(kk, :), N, 1);%NxD����Xshift��ĳһ������������£�ÿ����������������µĶԾ�ֵ������������(X-��i)
             pSigma(:, :, kk) = (Xshift' * ...
                 (diag(pGamma(:, kk)) * Xshift)) / Nk(kk);%DxD����pSigma(::,i) ����Pi  �������=1/Nk(i)
                 %��i���������= ��(X-��i)ת��*����Pi*(X-��i)*��i���������
         end
  
         % check for convergence
         L = sum(log(Px*pPi')); %���ϸ�˹�ֲ�����Ȼ����
        if L-Lprev < threshold %���ŵ������������ӣ���Ȼ����Խ��Խ��ֱ������
             break; %��Ȼ�����������˳�
         end
        Lprev = L;
     end
 
     if nargout == 1 %���������һ�������Ļ�����ôvarargout=Px;
         varargout = {Px};
     else %���򣬷���[Px model],����model�ǽṹ��
        model = [];
         model.Miu = pMiu;
         model.Sigma = pSigma;
         model.Pi = pPi;
         varargout = {Px, model};
     end
 

     function [pMiu pPi pSigma] = init_params()
         pMiu = centroids;%��X�����е�����K�У�KxD����  �����
         pPi = zeros(1, K);%��ȡKά������[0 0 ...0]     ��Ȩϵ����ÿ������������е���С����ĸ��ʣ�
         pSigma = zeros(D, D, K);%��ȡK��DxD�������
  
         %distmatΪDά�����ƽ����
         % hard assign x to each centroids  %X��NxD��sum(X.*X, 2)ΪNx1�� repmat(sum(X.*X, 2), 1, K)��������1������������K�������NxK
        distmat = repmat(sum(train_images.*train_images, 2), 1, K) + ... %distmat��j�еĵ�i��Ԫ�ر�ʾ��j���������i�������ľ��룬���������4��������2������ôdistmat����4*2����
            repmat(sum(pMiu.*pMiu, 2)', N, 1) - 2*train_images*pMiu'; %sum(A��2)���Ϊÿ���������������i��Ԫ���ǵ�i�е���ͣ�
        [dummy labels] = min(distmat, [], 2); %����������dummy��labels��dummy������¼distmat��ÿ�е���Сֵ��labels������¼ÿ����Сֵ���кţ����ȡ��һ���������ǵڼ������࣬labels��N��1��������NΪ������
 
         for k=1:K
             Xk = train_images(labels == k, :); %�ѱ�־Ϊͬһ������������������
            pPi(k) = size(Xk, 1)/N; %���ϸ�˹ģ�͵ļ�Ȩϵ����pPiΪ1*K������  
            pSigma(:, :, k) = cov(Xk); 
		%���XΪNx1���飬��ôcov(Xk)�󵥸���˹ģ�͵�Э�������
		%���XΪNxD(D>1)�ľ�����ôcov(Xk)�����������Э�������
		%cov()�����Ϊ����--��������������ͳ�ơ�-��ά����������������������ǶԳƾ��������Ǻ������ǶԳƣ�--�����Դ�����
		%pSigmaΪD*D*K�ľ���
        end
     end
  
     function Px = calc_prob()
         Px = zeros(N, K);%NxK�����
         for k = 1:K
             Xshift = train_images-repmat(pMiu(k, :), N, 1); %NxD����Xshift��ʾΪ����һ��1xD���������������N������������-Uk����i�б�ʾxi-uk
            inv_pSigma = inv(pSigma(:, :, k)); %��Э���������棬pSigmaD*D*K�ľ��� inv_pSigmaD*D�ľ���,��^-1
             tmp = sum((Xshift*inv_pSigma) .* Xshift, 2); 
     	%tmpΪN*1���󣬵�i�б�ʾ��xi-uk)^T*Sigma^-1*(xi-uk) ��-(x-��)ת��x(��^-1).(x-��)   ----�����в�ˣ�����ˣ��͵��
            coef = (2*pi)^(-D/2) * sqrt(det(inv_pSigma));
		 %���ά��̬�ֲ���ָ��ǰ���ϵ������2��)^(-d/2)*|(��^-(1/2))|
            Px(:, k) = coef * exp(-0.5*tmp); %NxK�����󵥶�һ��Dά��̬�ֲ����������ĸ����ܶȻ���
        end
     end
 end
%����  �����ھ�
%�����㷨 ���Դ���
%�����ܶȣ�����Ȼ  ����ͳ��