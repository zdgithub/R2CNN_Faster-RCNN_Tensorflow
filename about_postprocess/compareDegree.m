function [comp] = compareDegree(dt, gt)
% �������������б�ĶԱ���б�Ƕ�

i=1;
j=1;
comp = [];

while i<=size(dt,1)
    if j<=size(gt,1) && abs(dt(i,2)-gt(j,2))<10
        comp = [comp; dt(i,1),gt(j,1)];
        i = i+1;
        j = j+1;
    else
        if j>=size(gt, 1)
            i = i+1;
            j = 0;
        end
        j = j+1; 
    end
end


end
