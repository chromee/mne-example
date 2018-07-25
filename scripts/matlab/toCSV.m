root = "C:\Users\dk334\Downloads\eeg_raw_data\judo\mat\"
save_root = "C:\Users\dk334\Downloads\eeg_raw_data\judo\csv\"
files = dir(root)

for i = 1:numel(files)
    if not(files(i).isdir)
        path = root + files(i).name
        %disp(path)
        raw = load(path)
        y = raw.y
        yy = transpose(y)
        filename = save_root + files(i).name + ".csv"
        csvwrite(filename, yy)
    end
end
