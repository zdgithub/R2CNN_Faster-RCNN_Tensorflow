function contourRect(img)
% 在图上绘制人为标记的不规则四边形点的外接矩形

%img = 'L-14-5-16-31-04';
imgpath = ['E:\3AllRBox\VOCdevkit\VOCdevkit_train\JPEGImages\', img, '.jpg'];    %未旋转的原图
labelmat = ['E:\3AllRBox\VOCdevkit\VOCdevkit_train\relabelMat\', img, '.jpg.mat']; %人为标记点
%人为标记的不规则四边形坐标点x,y
load(labelmat, 'x', 'y'); 

box_x = [];
box_y = [];

hfig = figure('Visible', 'off');

for i=1:4:size(x,1)
    if i+3 > size(x,1)
        break;
    else
        x1 = x(i);   %左上角坐标
        y1 = y(i);
        x2 = x(i+1); %右上角坐标
        y2 = y(i+1);
        x3 = x(i+2); %右下角坐标
        y3 = y(i+2);
        x4 = x(i+3); %左下角坐标
        y4 = y(i+3);
        % 确定不规则四边形的外接矩形
        k = (y1-y2) / (x1-x2);
        b1 = y1 - k * x1;
        b3 = y3 - k * x3;
        b4 = y4 - k * x4;
        % 两平行直线间的距离
        Sh = abs(max(b3,b4) - b1) / sqrt(1+k*k);
        if k==0
            Sw = abs(min(x1,x4) - max(x2,x3));
            xx1 = min(x1,x4);
            yy1 = y1;
            xx3 = max(x2,x3);
            yy3 = max(y3,y4);
            xx2 = xx3;
            yy2 = yy1;
            xx4 = xx1;
            yy4 = yy3;
        else
            c1 = y1 + 1/k * x1;
            c4 = y4 + 1/k * x4;
            c2 = y2 + 1/k * x2;
            c3 = y3 + 1/k * x3;
            Sw = abs(max(c1,c4) - min(c2,c3)) / sqrt(1+1/(k*k));
            % 计算交点坐标
            xx1 = (max(c1,c4)-b1) / (k+1/k);
            yy1 = k * xx1 + b1;
            xx3 = (min(c2,c3) - max(b3,b4)) / (k+1/k);
            yy3 = k * xx3 + max(b3,b4);
            xx2 = (min(c2,c3)-b1) / (k+1/k);
            yy2 = k * xx2 + b1;
            xx4 = (max(c1,c4)-max(b3,b4)) / (k+1/k);
            yy4 = k * xx4 + max(b3,b4);
        end
        %排除nan数据
        newxy = [xx1, xx2, xx3, xx4, yy1, yy2, yy3, yy4];
        if any(isnan(newxy))==1
            disp(img);
            disp('exist nan');
            continue;
        end
        %周围扩大20%
        x_c = (xx1+xx3) / 2;
        y_c = (yy1+yy3) / 2;
        
        newx1 = 1.2*xx1 - 0.2*x_c;
        newy1 = 1.2*yy1 - 0.2*y_c;
        newx2 = 1.2*xx2 - 0.2*x_c;
        newy2 = 1.2*yy2 - 0.2*y_c;
        newx3 = 1.2*xx3 - 0.2*x_c;
        newy3 = 1.2*yy3 - 0.2*y_c;
        newx4 = 1.2*xx4 - 0.2*x_c;
        newy4 = 1.2*yy4 - 0.2*y_c;
 
        box_x = [box_x;newx1,newx2,newx3,newx4,newx1];
        box_y = [box_y;newy1,newy2,newy3,newy4,newy1];
    end
end
%save('box_y.mat', 'box_y')

%人为标记点
A = imread(imgpath);
% imshow(A);
% hold on;
% for i=1:4:size(x,1)
%     if i+3>size(x,1)
%         break;
%     else
%         tx = [x(i),x(i+1),x(i+2),x(i+3)];
%         ty = [y(i),y(i+1),y(i+2),y(i+3)];
%         plot(tx,ty,'r.');
%         hold on;
%     end
% end

%对人为标记点取外接矩形
imshow(A);
hold on;
for i=1:size(box_x,1)
    plot(box_x(i,:), box_y(i,:), 'r');
    hold on;
end

saveas(hfig, ['E:\3AllRBox\VOCdevkit\VOCdevkit_train\output\', img, '.jpg']);
    




