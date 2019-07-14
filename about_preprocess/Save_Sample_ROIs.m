function Save_Sample_ROIs()

r_path = 'E:\3AllRBox\VOCdevkit\VOCdevkit_train\';
folderpath = fullfile(r_path, 'tmp');
filepaths = dir(fullfile(folderpath, '*.jpg')); %列出该文件夹下所有.jpg格式的文件

%注意按照顺时针左上,右上,右下,左下的顺序标注每个骨节的四个顶点
for i = 1:length(filepaths)
    img = imread(fullfile([r_path , 'JPEGImages'], filepaths(i).name)); %读入第i个图片
    imshow(img);
    hold on;
    [x, y] = ginput; %调用matlab的ginput函数获取鼠标标记点的坐标值
    savePath = [r_path, 'relabelMat\', filepaths(i).name, '.mat'];
    save(savePath, 'x', 'y'); %将该张图上所点的坐标值保存到.mat文件中
    hold off;
end

end


