function [overlap_ratio] = single_IoU(bboxA,bboxB)

x1_A = bboxA(1);
y1_A = bboxA(2);

x2_A = x1_A + bboxA(3);
y2_A = y1_A + bboxA(4);

x1_B = bboxB(1);
y1_B = bboxB(2);

x2_B = x1_B + bboxB(3);
y2_B = y1_B + bboxB(4);

areaA = bboxA(3).*bboxA(4);
areaB = bboxB(3).*bboxB(4);

x1 = max(x1_A, x1_B); % inner coordinate x
y1 = max(y1_A, y1_B); % inner coordinate y
x2 = min(x2_A, x2_B); % outer coordinate x
y2 = min(y2_A, y2_B); % outer coordinate y

w = max(x2 - x1, 0);

h = max(y2 - y1, 0);

intersectAB = w*h;

overlap_ratio = intersectAB/(areaA+areaB - intersectAB);

end

