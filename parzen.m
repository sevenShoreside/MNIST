%  ��˹����Parzen ����ͳ������parzen���ڵĹ��Ƹ���
% train_images - ѵ������
%  h - ������
% test_images -��������
function p=parzen(train_images,h,test_images)
  p=zeros(1, size(train_images,2));
 % rows=size(train_images,1);
  for i=1: size(train_images,2)
    hn=h;
        p(i)=exp(-(test_images-train_images(:,i))'*(test_images-train_images(:,i))/(2*power(hn,2)))/(sqrt(2*pi)/(hn^25));
  end
end