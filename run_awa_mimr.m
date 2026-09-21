function result = run_awa_mimr()
%RUN_AWA_MIMR Reproduce the AWAphog HFS-MIMR experiment with CHA-CF.

project_root = fileparts(mfilename('fullpath'));
train_file = fullfile(project_root, 'datasets', 'AWAphogTrain.mat');
test_file = fullfile(project_root, 'datasets', 'AWAphogTest.mat');

result = run_relift_case( ...
    'AWAphog', train_file, test_file, 13, 0.25, 0.30);
end
