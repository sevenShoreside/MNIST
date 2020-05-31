%测试程序
function [error_label,error_rate,real_shibie]=Test(M,C,test_images,test_labels)
real_shibie=zeros(10,10);
error_label =zeros(1,size(test_labels,2));  
error_num = 0;
for i = 1:size(test_images,2)     
    max_p=0;
    for j=1:10                                       %最大似然估计
        if max_p < mvnpdf(test_images(:,i),M(:,j),C(:,:,j))
           max_p = mvnpdf(test_images(:,i),M(:,j),C(:,:,j));
           j_max = j;
        else
        end
    end
    if j_max == test_labels(i)+1         
    else
         error_num = error_num + 1;
         error_label(error_num) = i;
    end
     real_shibie(test_labels(i)+1,j_max)=real_shibie(test_labels(i)+1,j_max)+1;
end
%error_label(1,error_num+1:end) = [];
error_rate = size(error_label,2)/size(test_images,2);
end
 

