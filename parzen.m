%  高斯函数Parzen 窗，统计落在parzen窗内的估计概率
% train_images - 训练数据
%  h - 窗长度
% test_images -测试数据
function p=parzen(train_images,h,test_images)
  p=zeros(1, size(train_images,2));
 % rows=size(train_images,1);
  for i=1: size(train_images,2)
    hn=h;
        p(i)=exp(-(test_images-train_images(:,i))'*(test_images-train_images(:,i))/(2*power(hn,2)))/(sqrt(2*pi)/(hn^25));
  end
end