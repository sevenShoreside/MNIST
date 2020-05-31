%********����MNIST���ݿ����д������ʶ��ϵͳ*********
clear all;close all;clc;
[train_images,train_labels]=readMNIST('train-images.idx3-ubyte', 'train-labels.idx1-ubyte', 30000, 0);
size_image1 = size(train_images,1);
size_label1 = size(train_labels,1);
image_n(10).vector = [];
image_n_num = zeros(1,10);
for i=1:size_label1
    image_n_num(train_labels(i)+1) = image_n_num(train_labels(i)+1) + 1;
end
for i = 1:10
    image_n(i).vector = zeros(size(train_images,1),image_n_num(i));
end
image_n_num = zeros(1,10);
for i = 1:size_label1
    image_n_num(train_labels(i)+1) = image_n_num(train_labels(i)+1) + 1;
    image_n(train_labels(i)+1).vector(:,image_n_num(train_labels(i)+1)) = train_images(:,i);    
end
tic
%% ����˹ʶ�� %******ͨ��ѵ���ֱ�õ���ֵ��Э������������******************
C = zeros(size_image1,size_image1,10);%�����ʼ��
M = zeros(size_image1,10);%��ֵ��ʼ��
for i=1:10
    size_image2 = size(image_n(i).vector,2);
    M(:,i)=mean(image_n(i).vector,2);
    for j=1:size_image2
        C(:,:,i) = C(:,:,i) + (image_n(i).vector(:,j)-M(:,i))*(image_n(i).vector(:,j)-M(:,i))';
    end
    C(:,:,i) = C(:,:,i)/size_image2;
end
 %%**************************����***********************
[test_images,test_labels]=readMNIST('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', 10000, 0); %��ȡminist���ݿ⣬����������
[error_label,error_rate,real_shibie1] = Test(M,C,test_images,test_labels);
disp ����˹ʶ����Ϊ: 
disp(1-error_rate)
t1=toc;
tic

%% parzen��ʶ��   
real_shibie2=zeros(10,10); 
  error_label =zeros(1,size(test_labels,2));  
   error_num = 0;  
for i=1:size(test_images,2)
   p=parzen(train_images,1,test_images(:,i));%h����Ϊ1
   result=find(p==max(p));
    result=train_labels(result);
    if result == test_labels(i)         
    else
         error_num = error_num + 1;
         error_label(error_num) = i;
    end
     real_shibie2(test_labels(i)+1,result+1)=real_shibie2(test_labels(i)+1,result+1)+1; %ͳ��ÿ�����ֶ�Ӧ��ʶ������
end
error_rate = size(error_label,2)/size(test_images,2);
disp parzen��ʶ����Ϊ: 
disp(1-error_rate)
t2=toc;
tic
%% ���˹ʶ��
%ѵ��
class=3;  %�ֳɵĶ��˹����
for k=1:10
    data(class).vector=[];
    idx=kmeans(image_n(k).vector',class);
    c = zeros(25,25);
     mu=zeros(25,class);
    for i =1:class
        num(i)=length(find(idx==i));
        w1(i)=num(i)/image_n_num(k);
        data(i).vector=image_n(k).vector(:,find(idx==i));
        mu(:,i)=mean(data(i).vector,2);
        for j=1:num(i)
    %                 c=(data(1).vector(:,j)-mu1)*(data(1).vector(:,j)-mu1)';
            c=c+(data(i).vector(:,j)-mu(:,i)).*(data(i).vector(:,j)-mu(:,i))';
    %         c=(image_n(1).vector(:,find(idx==1),j)-mu1)*(image_n(1).vector(:,find(idx==1),j)-mu1)';
        end
        c1(:,:,i)=c/num(i);
    end
    disp('done!');
    GMM(k) = struct('mu',mu,'sigma',c1,'w',w1) ;
end
 %���Գ���
[test_images,test_labels]=readMNIST('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', 10000, 0);
real_shibie=zeros(10,10);
error_label =zeros(1,size(test_labels,2));  
error_num = 0;
for i=1:size(test_images,2)
    max_p=0;
    for j=1:10
        %w(j)=length(find(train_labels==(j-1)))/size(train_images,2);
        px=0;                                        
        for k=1:class
            px=px+GMM(j).w(k)*mvnpdf(test_images(:,i),GMM(j).mu(:,k),GMM(j).sigma(:,:,k));
        end
        if max_p<px
           max_p=px 
           jmax=j;
        end
    end
    if jmax == test_labels(i)+1         
    else
         error_num = error_num + 1;
         error_label(error_num) = i;
    end
    real_shibie(test_labels(i)+1,jmax)=real_shibie(test_labels(i)+1,jmax)+1;
end
error_label(1,error_num+1:end) = [0];
error_rate = size(error_label,2)/size(test_images,2);
disp ���˹ʶ����Ϊ: 
disp(1-error_rate)
disp('done')
t3=toc;

 

