function mat2xml(imgpath, savepath, savefile, x, y)
% 将[x, y]标签保存为xml格式的标签

%create document
docNode = com.mathworks.xml.XMLUtils.createDocument('annotation');
%document element
docRootNode = docNode.getDocumentElement();
% folder
folderNode = docNode.createElement('folder');
folderNode.appendChild(docNode.createTextNode('XrayData'));
docRootNode.appendChild(folderNode);
% filename
savename = [savefile, '.jpg'];
A = imread(fullfile(imgpath, savename));
filenameNode = docNode.createElement('filename');
filenameNode.appendChild(docNode.createTextNode(savename));
docRootNode.appendChild(filenameNode);
% source 文件来源不重要
sourceElement = docNode.createElement('source');
docRootNode.appendChild(sourceElement);

databaseNode = docNode.createElement('database');
databaseNode.appendChild(docNode.createTextNode('Xray Database'));
sourceElement.appendChild(databaseNode);

annotationNode = docNode.createElement('annotation');
annotationNode.appendChild(docNode.createTextNode('Xray Datas'));
sourceElement.appendChild(annotationNode);

imageNode = docNode.createElement('image');
imageNode.appendChild(docNode.createTextNode('xrays'));
sourceElement.appendChild(imageNode);

flickridNode = docNode.createElement('flickrid');
flickridNode.appendChild(docNode.createTextNode('NULL'));
sourceElement.appendChild(flickridNode);
% owner 文件作者不重要
ownerElement = docNode.createElement('owner');
docRootNode.appendChild(ownerElement);

flickridNode = docNode.createElement('flickrid');
flickridNode.appendChild(docNode.createTextNode('NULL'));
ownerElement.appendChild(flickridNode);

nameNode = docNode.createElement('name');
nameNode.appendChild(docNode.createTextNode('hospital'));
ownerElement.appendChild(nameNode);
% size 图像尺寸
sizeElement = docNode.createElement('size');
docRootNode.appendChild(sizeElement);
[h, w, c] = size(A);
widthNode = docNode.createElement('width');
widthNode.appendChild(docNode.createTextNode(num2str(w)));
sizeElement.appendChild(widthNode);

heightNode = docNode.createElement('height');
heightNode.appendChild(docNode.createTextNode(num2str(h)));
sizeElement.appendChild(heightNode);

depthNode = docNode.createElement('depth');
depthNode.appendChild(docNode.createTextNode(num2str(c)));
sizeElement.appendChild(depthNode);
% segmented 是否用于分割
segmentedNode = docNode.createElement('segmented');
segmentedNode.appendChild(docNode.createTextNode('0'));
docRootNode.appendChild(segmentedNode);
% object 检测目标
n = size(x,1);
for i=1:4:n
    if i+3>n
        break;
    end
    objElement = docNode.createElement('object');
    docRootNode.appendChild(objElement);

    objNameNode = docNode.createElement('name');
    objNameNode.appendChild(docNode.createTextNode('bone')); %检测对象/类别
    objElement.appendChild(objNameNode);

    poseNode = docNode.createElement('pose');
    poseNode.appendChild(docNode.createTextNode('Unspecified')); %拍摄角度
    objElement.appendChild(poseNode);

    truncatedNode = docNode.createElement('truncated');
    truncatedNode.appendChild(docNode.createTextNode('0')); %是否被截断
    objElement.appendChild(truncatedNode);

    difficultNode = docNode.createElement('difficult');
    difficultNode.appendChild(docNode.createTextNode('0')); %是否容易被检测
    objElement.appendChild(difficultNode);

    bboxElement = docNode.createElement('bndbox');
    objElement.appendChild(bboxElement);
%     xmin = x(i);
%     xmax = x(i+1);
%     ymin = y(i);
%     ymax = y(i+1);
%     wlen = abs(xmax-xmin);
%     hlen = abs(ymax-ymin);
%     xmin = xmin - wlen*0.05;
%     xmax = xmax + wlen*0.05;
%     ymin = ymin - hlen*0.05;
%     ymax = ymax + hlen*0.05;
%     
%     xminNode = docNode.createElement('xmin');
%     xminNode.appendChild(docNode.createTextNode(num2str(xmin)));
%     bboxElement.appendChild(xminNode);
%     
%     yminNode = docNode.createElement('ymin');
%     yminNode.appendChild(docNode.createTextNode(num2str(ymin)));
%     bboxElement.appendChild(yminNode);
%     
%     xmaxNode = docNode.createElement('xmax');
%     xmaxNode.appendChild(docNode.createTextNode(num2str(xmax)));
%     bboxElement.appendChild(xmaxNode);
%     
%     ymaxNode = docNode.createElement('ymax');
%     ymaxNode.appendChild(docNode.createTextNode(num2str(ymax)));
%     bboxElement.appendChild(ymaxNode);
%将人为标记的不规则四边形转变成规则的外接矩形
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
    if k==0
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
        disp(savefile);
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
    
    x0Node = docNode.createElement('x0');
    x0Node.appendChild(docNode.createTextNode(num2str(newx1)));
    bboxElement.appendChild(x0Node);

    y0Node = docNode.createElement('y0');
    y0Node.appendChild(docNode.createTextNode(num2str(newy1)));
    bboxElement.appendChild(y0Node);

    x1Node = docNode.createElement('x1');
    x1Node.appendChild(docNode.createTextNode(num2str(newx2)));
    bboxElement.appendChild(x1Node);

    y1Node = docNode.createElement('y1');
    y1Node.appendChild(docNode.createTextNode(num2str(newy2)));
    bboxElement.appendChild(y1Node);
    
    x2Node = docNode.createElement('x2');
    x2Node.appendChild(docNode.createTextNode(num2str(newx3)));
    bboxElement.appendChild(x2Node);

    y2Node = docNode.createElement('y2');
    y2Node.appendChild(docNode.createTextNode(num2str(newy3)));
    bboxElement.appendChild(y2Node);
    
    x3Node = docNode.createElement('x3');
    x3Node.appendChild(docNode.createTextNode(num2str(newx4)));
    bboxElement.appendChild(x3Node);

    y3Node = docNode.createElement('y3');
    y3Node.appendChild(docNode.createTextNode(num2str(newy4)));
    bboxElement.appendChild(y3Node);
end


% xml write
xmlname = [savepath, '\', savefile, '.xml'];
xmlwrite(xmlname, docNode);
imwrite(A,['E:\3AllRBox\VOCdevkit\VOCdevkit_train\output\',savename]);
