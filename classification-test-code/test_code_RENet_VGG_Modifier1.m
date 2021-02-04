format compact; clear; clc;

model_path = 'models/net_RENet_VGG_Modified1.mat';
if isfile(model_path)
    fprintf("Model loading...\n");
    load(model_path,'net_RENet_VGG_Modified_1');
    fprintf("Model loaded successfully.\n");
    [filepath, cancel] = imgetfile();
    if (~cancel)
        original_image = imread(filepath);
        resized_image = imresize(original_image,[224 224]);
        channels = size(size(resized_image));
        if channels(2) == 2
            resized_image = cat(3, resized_image, resized_image, resized_image);
        end
        disp(size(resized_image));
        fprintf("Running Model on Image.\n");
        [predicted_labels,posterior] = classify(net_RENet_VGG_Modified_1,resized_image);
        fprintf("posterior: %f\n", posterior);
        fprintf("predicted_labels: %f\n", predicted_labels);
        label = categorical(predicted_labels);
        if (label == "COVID-19")
            fprintf(' \n COVID-19 Infected \n')
        else
            fprintf(' \n Healthy \n')
        end
    else
        fprintf("Invalid Image selected.\n");
    end
else
    fprintf("Model could not be loaded.\n");
end
