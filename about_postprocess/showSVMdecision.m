function showSVMdecision()
p = './nowSVM/';
files = dir(fullfile([p, 'R2CNN_rotatebox_2150_new/'], '*.jpg'));
%img = 'L-20-11-35-35';
mkdir([p, 'shownow']);

for i=1:length(files)
    hfig = figure('Visible', 'off');
    img = files(i).name(1:end-10);
    A = imread([p, 'R2CNN_rotatebox_2150_new/', files(i).name]);
    imshow(A);
    hold on;
    % dtbox= [x0, y0, x1, y1, x2, y2, x3, y3, degree, ycenter, 1abel];
    load([p, 'dtboxSet/', img, '.mat'], 'dtbox');
    load([p, 'testpred/', img, '-pred.mat'], 'pred');
    n = length(pred);
    idx = [2:(n+1)]; %不含首尾的骨节
    decision = [idx', pred'];
    decision = sortrows(decision, 2); %将预测值从小到大排序
    
    for k=1:size(decision,1)
        j = decision(k,1); %椎骨编号
        pre = decision(k,2); %椎骨预测值
        if pre>0  %只画出负样本的概率
            break;
        end
        x0 = dtbox(j,1);
        y0 = dtbox(j,2);
        x1 = dtbox(j,3);
        y1 = dtbox(j,4);
        x2 = dtbox(j,5);
        y2 = dtbox(j,6);
        x3 = dtbox(j,7);
        y3 = dtbox(j,8);
        xcenter = (x0+x2)/2;
        ycenter = (y0+y2)/2;
        text(double(xcenter), double(ycenter), num2str(pre),'Color','r');
    end  
    saveas(hfig, [p, 'shownow/', img, '.jpg']);
end



