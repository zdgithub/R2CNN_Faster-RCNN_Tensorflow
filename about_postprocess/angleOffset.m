% compute the different angle-offset between dt boxes and gt boxes
p1 = 'gtbox';
p2 = 'dtbox';
files = dir(fullfile(p1, '*.mat'));
bias = [];
for k=1:length(files)
    gtp = ['gtbox\', files(k).name];
    load(gtp, 'gtbox');  %x1, y1, x2, y2, theta, up_yc
    dtp = ['dtbox\', files(k).name(1:end-4), '.jpg.mat'];
    load(dtp, 'dtbox');  %x1, y1, x2, y2, h, theta, up_yc
    gt = gtbox(:, end-1:end);
    dt = dtbox(:, end-1:end);
    diff = compareDegree(dt, gt);
    %bias = [];
    for i=1:size(diff, 1)
        for j=i+1:size(diff, 1)
            dt_d = diff(i,1) - diff(j,1);
            gt_d = diff(i,2) - diff(j,2);
            tmp = dt_d - gt_d;
            bias = [bias, tmp];
        end
    end
    if abs(max(bias)) >= 10
        disp(files(k).name)
    end
    %savep = ['diff\', files(k).name, '.mat'];
    %save(savep, 'bias');
end
bias = abs(bias);
mean_val = mean(bias);
std_val = std(bias);
figure;
hist(bias);
title(['mean=',num2str(mean_val),', std=', num2str(std_val)]);

% x = -15:3:15;
% y = zeros(1,length(x));
% for z = 1:length(x)
%     y(z) = length(find(bias==x(z)));
% end
% bar(x, y)
% for i=1:length(x)
%     text(x(i), y(i), num2str(y(i),'%g'),...
%     'HorizontalAlignment', 'center',...
%     'VerticalAlignment', 'bottom')
% end


% file = 'test-102.mat';
% gtp = ['gtbox\', file];
% load(gtp, 'gtbox');  %x1, y1, x2, y2, theta, up_yc
% dtp = ['dtbox\', file(1:end-4), '.jpg.mat'];
% load(dtp, 'dtbox');  %x1, y1, x2, y2, h, theta, up_yc
% gt = gtbox(:, end-1:end);
% dt = dtbox(:, end-1:end);
% diff = compareDegree(dt, gt);
% bias = [];
% for i=1:size(diff, 1)
%     for j=i+1:size(diff, 1)
%         dt_d = diff(i,1) - diff(j,1);
%         gt_d = diff(i,2) - diff(j,2);
%         tmp = dt_d - gt_d;
%         if abs(tmp)>=10
%             disp([num2str(i),'-', num2str(j)])
%         end
%         bias = [bias, tmp];
%     end
% end

        
        
    


