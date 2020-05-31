function digits = trimDigits1(digitsIn)
    dSize = size(digitsIn);
    digits = zeros(5*5,dSize(3));
    for i=1:dSize(3)
        for j=1:5
           for k=1:5
               digits_ = digitsIn((j-1)*4+1:j*4, (k-1)*4+1:k*4, i);
               digits((j-1)*5+k,i) = sum(digits_(:));  
           end
        end
    end        
end


