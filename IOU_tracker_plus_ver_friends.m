%=========================================================================%
% IOU tracker +                                                           %
% Written by Guisik Kim at XXXXXX xx, 2018.                               %
% Computer Vision Machine Learning lab., CAU, Seoul, Korea.               %
%                                                                         % 
%                                                                         % 
%                                                                         % 
% note :                                                                  % 
% inputs are Friends video sequence and detection file (txt).             %
%                                                                         %
% functions :                                                             %
% single_IoU.m (IOU)                                                      %
%=========================================================================%

clear;
close all;
clc;

warning('off');

time = 0;

% Parameters for Friends
iou_threshold = 0.6;            % Friends2
matching_threshold = 100;
galleries_num = 3;
distance_threshold = 300;


addpath('./friends2\');

det = load('friends2_gt.txt');

[~, idx] = sort(det(:, 1));

det = det(idx, :);

gt_data = det(:, 3 : 6);

gt_data = [det(:, 1) gt_data];

gt_data = [gt_data; ones(1 , 5)];

img_files = dir('./friends2\\*.jpg');
%--------------------------------------------------------------------------

%% People name

% name_list{1} = 'Joey';
% name_list{2} = 'Monica';
% name_list{3} = 'Phoebe';
% name_list{4} = 'Ross';

%%

% Make BBox
for i = 1 : 100
    label_color{i} = [rand(1, 1)  rand(1, 1)  rand(1, 1)];
end

% Count object number
count = 1;

temp_gt = gt_data;

for current_frame = 1 : numel(gt_data(:, 1)) - 1
        
    if sum(gt_data(current_frame, 2 : 5)) ~= 0
    temp_value = gt_data(current_frame, 1) - gt_data(current_frame + 1, 1);
    
    count = count + 1;

    if temp_value ~= 0
        
        object_num(current_frame) = count;
                
        count = 1;
        
    end
    
    else
        object_num(current_frame) = 1;
        temp_gt(current_frame, :) = 0;
    end
    
end

temp_gt = temp_gt(temp_gt > 0);

gt_data = reshape(temp_gt, [], 5);

object_num = nonzeros(object_num) - 1;

accum_num = 0;

score_data = det(:, end);

max_id = 0;

tracker_results = zeros(1, 10);

previous_data = [];

temp_error = 0;

% Algorithm
%--------------------------------------------------------------------------
for current_frame = 1 : numel(img_files)
    
    img = imread(img_files(current_frame).name);
        
    current_gt = gt_data(accum_num + 1 : accum_num + object_num(current_frame), 2 : end);
        
    current_score = score_data(accum_num + 1 : accum_num + object_num(current_frame));
    
    current_id = 1 : object_num(current_frame);
    
    accum_num = accum_num + object_num(current_frame);
                
    if sum(current_gt) == 0
       current_gt = []; 
    end
    
    figure(1),
    tic()
    
    if current_frame == 1
        
        im_handle = imshow(img, 'Border', 'tight', 'InitialMag', 100);
        
        for i = 1 : object_num(current_frame)
            rect_handle{i} = rectangle('Position', current_gt(i, :), 'LineStyle', '-', 'LineWidth', 2, 'EdgeColor', label_color{i});
            id_handle{i} = text(current_gt(i, 1) - 10, current_gt(i, 2) - 10, num2str(current_id(i)), 'color', label_color{i}, 'fontsize', 15);
        end
        
        text_handle = text(10, 10, int2str(current_frame));
        set(text_handle, 'color', [1 1 0]);
        
    else
                
        set(im_handle, 'CData', img);
        set(text_handle, 'string', int2str(current_frame));
                           
        overlapRatio = [];
        idx = [];
        
        if ~isempty(current_gt) && ~isempty(tracks_active)
            
        % IOU matrix
        for r = 1 : numel(tracks_active(:, 1))
            for c = 1 : numel(current_gt(:, 1))
                
                overlapRatio(r, c) = single_IoU(tracks_active(r, 1 : 4), current_gt(c, :));
                
            end
        end

        projection_score = sum(overlapRatio > iou_threshold, 2);
        
        duple_idx = find(projection_score > 1);
        
        [~, sec_idx] = sort(overlapRatio(duple_idx, :)');
                                
        idx_nnz = max(overlapRatio > iou_threshold)';
                
        [~, b] = max(overlapRatio);
                
        if ~isempty(duple_idx)
            for i = 1 : numel(duple_idx)
                 
                temp_b = b.*idx_nnz;
                
                [w, e] = sort(overlapRatio(duple_idx(i), :));
                
                col_idx = e(1, 1 : end-1);
                
                overlapRatio(duple_idx(i), col_idx) = 0;
                
            end
        end
        
        if ~isempty(overlapRatio)
            
            if numel(overlapRatio(:, 1)) < 2
                
                temp_matrix = zeros(1, numel(overlapRatio));
                
                overlapRatio = [overlapRatio; temp_matrix];
                
            end
            
            idx_nnz = max(overlapRatio > iou_threshold);
            
            [a, b] = max(overlapRatio);
            
            temp_id = idx_nnz .* tracks_active(b, 5)';
            
            [~, idx] = find(temp_id == 0);
        
        if ~isempty(idx)
            
            distance = [];
            current_distance = [];
            
            % New ID assignment strategy
            %--------------------------------------------------------------
            new_id_candidate = current_gt(idx, :);
            
            for i = 1 : length(idx)
                distance(i, :) = sqrt((new_id_candidate(i, 1) - previous_appearance(:, 1)).^2 + (new_id_candidate(i, 2) - previous_appearance(:, 2)).^2);
                current_distance(i, :) = sqrt((new_id_candidate(i, 1) - current_gt(:, 1)).^2 + (new_id_candidate(i, 2) - current_gt(:, 2)).^2);
            end
            
            [~, current_distance_idx] = sort(current_distance');
            [distance_value, distance_idx] = sort(distance');
                                        
            new_id_appearance = [];
            nearest_appearance = [];
            similarity_error = [];
            
            for new_id_num = 1 : length(new_id_candidate(:, 1))
                    
                if numel(current_distance_idx(:, 1)) > 1 && current_distance(new_id_num, current_distance_idx(2, new_id_num)) > distance_threshold
                    
                    temp_id(idx(new_id_num)) = max(max(previous_id), max_id) + 1;
                    
                    max_id = max(temp_id);
                    
                else
                                        
                    new_id_appearance{new_id_num} = rgb2gray(imcrop(img, new_id_candidate(new_id_num, :)));
                    
                    if numel(distance_idx(:, new_id_num)) < galleries_num
                        
                        if numel(previous_appearance(:, 1)) == numel(distance_idx(:, 1))
                           
                            for i = 1 : numel(distance_idx(:, 1))

                                nearest_appearance{i} = galleries{distance_idx(i, new_id_num)};
                                nearest_appearance{i} = imresize(nearest_appearance{i}, [size(new_id_appearance{new_id_num}, 1), size(new_id_appearance{new_id_num}, 2)]);
                                
                                similarity_error(i) = (mean(mean(abs(new_id_appearance{new_id_num}(:) - nearest_appearance{i}(:)))));
                                
                            end
                        end
                        
                    else
                        
                        for i = 1 : galleries_num

                            nearest_appearance{i} = galleries{distance_idx(i, new_id_num)};
                            nearest_appearance{i} = imresize(nearest_appearance{i}, [size(new_id_appearance{new_id_num}, 1), size(new_id_appearance{new_id_num}, 2)]);
                            
                            similarity_error(i) = (mean(mean(abs(new_id_appearance{new_id_num}(:) - nearest_appearance{i}(:)))));
                                                        
                        end
                        
                    end
                    
                    distance_weight = distance_value(1 : numel(similarity_error), new_id_num)';
                    distance_weight = distance_weight./sum(distance_weight);

                    weighted_sum = similarity_error.*distance_weight;
                    
%                     temp_error = [temp_error; weighted_sum];
                    
                    [~, min_error_idx] = min(weighted_sum);

                    if weighted_sum(min_error_idx) <= matching_threshold
                        
                        if find(temp_id == previous_appearance(distance_idx(min_error_idx, new_id_num), 5))
                            
                            temp_id(idx(new_id_num)) = max(max(previous_appearance(:, 5)), max_id) + 1;
                            
                            max_id = max(temp_id);
                            
                        else
                            temp_id(idx(new_id_num)) = previous_appearance(distance_idx(min_error_idx, new_id_num), 5);
                        end                        
                    else
                        
                        temp_id(idx(new_id_num)) = max(max(previous_appearance(:, 5)), max_id) + 1;
                        
                        max_id = max(temp_id);
                        
                    end
                    
                end
                
            end
            %--------------------------------------------------------------
        end

        end
        
        current_id = temp_id;
        
        current_data = [current_gt current_id'];
                   
        new_id_set = current_data(idx, :);

        if flag == 1
            for i = 1 : numel(rect_handle)
                delete(rect_handle{i})
                delete(id_handle{i})
            end
        end
        
        for i = 1 : object_num(current_frame)
            
            rect_handle{i} = rectangle('Position', current_gt(i, :), 'LineStyle', '-', 'LineWidth', 5, 'EdgeColor', label_color{current_id(i)});
            id_handle{i} = text(current_gt(i, 1) - 10, current_gt(i, 2) - 10, num2str(current_id(i)), 'color', label_color{current_id(i)}, 'fontsize', 15);
            flag = 1;
        end
          
         previous_center_position = [tracks_active(:, 1) + tracks_active(:, 3)/2, tracks_active(:, 2) + tracks_active(:, 4)/2, previous_id'];
         current_center_position = [current_gt(:, 1) + current_gt(:, 3)/2, current_gt(:, 2) + current_gt(:, 4)/2, current_id'];

         trajectory_idx = [];

        end
    end

    previous_gt = current_gt;
    previous_id = current_id;
    
    tracks_active = [];
    
    if ~isempty(current_gt)

    previous_data = [current_gt current_id'];

    tracks_active = previous_data;
    
    [~, sort_tracks_active_idx] = sort(tracks_active(:, 5));
        
    tracks_active = tracks_active(sort_tracks_active_idx, :);
    
    max_active_value = max(tracks_active(:, 5));
    
    % Adaptive gallery update
    %----------------------------------------------------------------------
    if current_frame == 1
        
        previous_appearance = tracks_active;
        previous_img = rgb2gray(img);

        for i = 1 : numel(previous_appearance(:, 1))
            galleries{i} = imcrop(previous_img, previous_appearance(i, 1 : 4));
            c_galleries{i} = imcrop(img, previous_appearance(i, 1 : 4));
        end
        
         for i = 1 : numel(rect_handle)
                delete(rect_handle{i})
                delete(id_handle{i})
         end
    else
        
        previous_appearance(tracks_active(sort_tracks_active_idx, 5), :) = tracks_active(sort_tracks_active_idx, :);
        
        if previous_appearance(end, 5) < max_active_value
            
            previous_appearance(end + 1, :) = tracks_active(max_active_value : end, :);
            
        end

        previous_img = rgb2gray(img);
        
        update_list = tracks_active(sort_tracks_active_idx, 5);
        for i = 1 : numel(update_list)
            galleries{update_list(i)} = imcrop(previous_img, previous_appearance(update_list(i), 1 : 4));
            c_galleries{update_list(i)} = imcrop(img, previous_appearance(update_list(i), 1 : 4));
        end
        
    end
    
    else
        if ~isempty(previous_data)
        tracks_active = previous_data;
        end
    end
    %----------------------------------------------------------------------

    time = time + toc();
    
    % To output MOT results
    if ~isempty(current_gt)
        garbage_data = zeros(numel(tracks_active(:, 1)), 1) - 1;
        object_number = ones(numel(tracks_active(:, 1)), 1) * current_frame;
        tracker_results = [tracker_results; object_number tracks_active(:, 5) tracks_active(:, 1 : 4) current_score garbage_data garbage_data garbage_data];
    end
    
end

end_track = tracker_results(2 : end, :);

