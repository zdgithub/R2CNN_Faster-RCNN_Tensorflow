
% fpath = 'E:\trainset\label';
% files = dir(fullfile(fpath, '*.mat'));
% for i=1:length(files)
%     p = fullfile(fpath, files(i).name);
%     load(p, 'labels');
%     if sum(sum(isnan(fuv))) > 0
%         disp(files(i).name);
%         disp('nan exists');
%     end
% end

%data = textread('loss.txt');
% ������Ч����λ��
%vpa(a, 6);


% ������Ϊ��ע�����Ӿ��Σ�����label���Ƿ����׼ȷ
imgpath = 'E:\3AllRBox\VOCdevkit\VOCdevkit_train\tmp';
files = dir(fullfile(imgpath, '*.jpg'));
for i=1:length(files)
    imgname = files(i).name(1:end-4);
    contourRect(imgname);
end


% delpath = 'E:\3AllRBox\VOCdevkit\VOCdevkit_train\preweight';
% files = dir(fullfile(delpath, '*.jpg'));
% for i=1:length(files)
%     img = ['E:\3AllRBox\VOCdevkit\VOCdevkit_train\output\' ,files(i).name];
%     delete(img);
% end

% imgpath = 'C:\Users\lenovo\Desktop\resWrong';
% files = dir(fullfile(imgpath, '*.jpg'));
% for i=1:length(files)
%     imgname = files(i).name(1:end-6);
%     srcpath=['C:\Users\lenovo\Desktop\testUnlabel\', imgname];
%     despath=['C:\Users\lenovo\Desktop\testWrong\', imgname];
%     copyfile(srcpath, despath);
% end










    
    



