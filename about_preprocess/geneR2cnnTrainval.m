function geneR2cnnTrainval()
%生成VOC的训练集格式
clear;
imgpath = 'E:\3AllRBox\VOCdevkit\VOCdevkit_train\JPEGImages\';
matpath = 'E:\3AllRBox\VOCdevkit\VOCdevkit_train\relabelMat\';

filepath='E:\3AllRBox\VOCdevkit\VOCdevkit_train\relabel';
files = dir(fullfile(filepath, '*.jpg'));

savepath='E:\3AllRBox\VOCdevkit\VOCdevkit_train\relabelAnnotation';

for i=1:length(files)
    mat = [matpath, files(i).name(1:end-6), '.mat'];
    load(mat, 'x', 'y');
    tmp = files(i).name(1:end-10);
    mat2xml(imgpath, savepath, tmp, x, y);
end
%------------------------------------------------------
%分出一部分当验证集
%{
xmlpath = 'E:\1RBox\VOCdevkit\VOCdevkit_train\Annotation\';
imgpath = 'E:\1RBox\VOCdevkit\VOCdevkit_train\JPEGImages\';
valxml = 'E:\1RBox\VOCdevkit\VOCdevkit_test\Annotation\';
valimg = 'E:\1RBox\VOCdevkit\VOCdevkit_test\JPEGImages\';

xmlfiles = dir(fullfile(xmlpath, '*.xml'));
setsize = length(xmlfiles);
val_percent = 0.05;
valset = sort(randperm(setsize, floor(setsize*val_percent)));

for i=1:setsize
    if ismember(i, valset)
        movefile([xmlpath, xmlfiles(i).name], valxml);
        movefile([imgpath, xmlfiles(i).name(1:end-4), '.jpg'], valimg);
    end
end
%}
%{
prepath = 'E:\LeftData\21-30\';
nowpath = 'E:\LeftData\21-30-two\';
files = dir(fullfile(prepath, '*.jpg'));
n = length(files);
mv_percent = 0.5;
mvset = sort(randperm(n, floor(n*mv_percent)));

for i=1:n
    if ismember(i, mvset)
        movefile([prepath, files(i).name], nowpath);
    end
end
%}






