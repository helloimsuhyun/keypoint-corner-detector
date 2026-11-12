function run_ahmed(input_file, output_file, adapted_dir, level, seed)
% Author algorithms run in metres; only dataset I/O and initial sampling live here.
addpath(adapted_dir, '-begin');
rng(seed, 'twister');
C = readmatrix(input_file, 'FileType', 'text');
if size(C, 2) ~= 3 || any(~isfinite(C(:))) || size(C, 1) < 400
    error('Ahmed:Input', 'Expected at least 400 finite XYZ points in metres.');
end
% Same initial sampling as Corner_detection_main.m. Its workpiece-specific
% passthrough bounds and x==0 deletion do not apply to our saved ROI/full input.
Cloud = pointCloud(C);
ptCloudA = pcdownsample(Cloud, 'gridAverage', 0.01);
C = ptCloudA.Location;
if size(C, 1) < 400
    error('Ahmed:Support', 'Author 400-NN stage needs >=400 points after 10 mm grid averaging.');
end
[edge_points, edge_id, curve, surface_normal, cloud] = edgedetection(C);
predictions = corner_detection(cloud, edge_id, edge_points, curve, surface_normal, [], []);
selected = predictions{level};
fid = fopen(output_file, 'w');
if fid == -1
    error('Ahmed:Output', 'Cannot open prediction output.');
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'x_m,y_m,z_m\n');
fprintf(fid, '%.17g,%.17g,%.17g\n', selected');
save(fullfile(fileparts(output_file), 'ahmed_all_levels.mat'), 'predictions', 'level', 'seed');
end
