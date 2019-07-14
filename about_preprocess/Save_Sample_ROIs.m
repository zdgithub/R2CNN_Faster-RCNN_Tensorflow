function Save_Sample_ROIs()

r_path = 'E:\3AllRBox\VOCdevkit\VOCdevkit_train\';
folderpath = fullfile(r_path, 'tmp');
filepaths = dir(fullfile(folderpath, '*.jpg')); %�г����ļ���������.jpg��ʽ���ļ�

%ע�ⰴ��˳ʱ������,����,����,���µ�˳���עÿ���ǽڵ��ĸ�����
for i = 1:length(filepaths)
    img = imread(fullfile([r_path , 'JPEGImages'], filepaths(i).name)); %�����i��ͼƬ
    imshow(img);
    hold on;
    [x, y] = ginput; %����matlab��ginput������ȡ����ǵ������ֵ
    savePath = [r_path, 'relabelMat\', filepaths(i).name, '.mat'];
    save(savePath, 'x', 'y'); %������ͼ�����������ֵ���浽.mat�ļ���
    hold off;
end

end


