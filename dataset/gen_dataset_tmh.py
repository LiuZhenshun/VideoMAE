import os
import random
import subprocess
import openpyxl
from threading import Thread, Semaphore


class genDataset:
    def __init__(self, labelFile) -> None:
        self.classNames = ["normal", "self_extubated", "rail_lowered", "agitated"]
        self.processedAnns = self.process_labelFile(labelFile)
    

    def process_labelFile(self, path):
        # Load the workbook
        workbook = openpyxl.load_workbook(path)
        # Get the first worksheet (you can also select a specific sheet by name)
        sheet = workbook.active

        # Iterate through rows in the sheet
        rows = []
        for row in sheet.iter_rows():
            # Read the values of each cell in the row and store them in a list
            row_values = [cell.value for cell in row]
            # Check if the row is empty and skip it if it is
            if all(value is None or (isinstance(value, str) and value.strip() == '') for value in row_values):
                continue
            rows.append(row_values)
        rows = rows[1:]

        processedAnn = []
        for index1 in range(0, len(rows), 3):
            elementRow = {}
            for index2 in range(3):
                timeAnnos = []
                for index3,element in enumerate(rows[index1+index2]):
                    if element == None:
                        elementRow[self.classNames[index2 + 1]] = timeAnnos
                        break
                    if index3 == 0:
                        elementRow["video_name"] = element
                        continue
                    elif index3 == 1:
                        continue
                    timeAnnos.append(element)
            if elementRow['video_name'].find('_sides') != -1:
                elementRow['video_name'] = elementRow['video_name'].replace('_sides', '')
                processedAnn.append(self.find_overlaps(elementRow))

        return processedAnn
    

    def find_overlaps(self, data):
        # Convert all the timestamps into seconds and create ranges
        ranges = {c: list(zip(data[c][::2], data[c][1::2])) for c in data if c != 'video_name'}

        # Convert timestamp ranges into seconds
        ranges_in_seconds = {c: [(self.timestamp_to_seconds(start), self.timestamp_to_seconds(end)) for start, end in ranges[c]] for c in ranges}

        # Flatten the list of all labeled intervals
        labeled_intervals = [(start, end, label) for label in ranges_in_seconds for start, end in ranges_in_seconds[label]]

        # Sort by start time
        labeled_intervals.sort()

        # Merge intervals
        merged = self.extract_overlaps(labeled_intervals)

        return {'video_name': data['video_name'], 'label': merged}
    

    def cut_videos(self, classPath, dstClassPath, maxThreads=300):
        threads = []
        semaphore = Semaphore(maxThreads)

        for processedAnn in self.processedAnns:
            
            video_folder = os.path.splitext(processedAnn['video_name'])[0]

            # Process the labeled time intervals
            for labelId, label in enumerate(processedAnn['label']):
                os.makedirs(os.path.join(dstClassPath, video_folder), exist_ok=True)
                dstVideoPath = os.path.join(dstClassPath, video_folder, (label[2] + f"{labelId + 1}" + ".mp4"))               
                thread = Thread(target=self.vid2jpg,
                                args=(processedAnn['video_name'], classPath, dstVideoPath, label[0], label[1], semaphore))
                thread.start()
                threads.append(thread)

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

    
    def gen_label(self, imgFolder):
        outLabelPath = os.path.join(imgFolder, "labels")
        os.makedirs(outLabelPath, exist_ok=True)
        output = []
        videoNames= os.listdir(imgFolder)
        class_name = {"normal":0, "self_extubated":1, "rail_lowered":2, "agitated":3}

        for videoName in videoNames:
            classNames = os.listdir(os.path.join(imgFolder,videoName))
            for className in classNames:
                classPath = os.path.join(imgFolder,videoName,className)
                label = ""
                for name in self.classNames:
                    if className.find(name) != -1:
                        label = label + str(class_name[name])
                output.append('%s %s'%(classPath, label[0]))
        # Set the random seed so the results are reproducible
        random.seed(42)
        # Shuffle the list
        random.shuffle(output)
        # Calculate the index where to split the dataset
        train_end_index = int(len(output) * 0.8)
        val_end_index = train_end_index + int(len(output) * 0.2) // 2
        # Split the dataset into training (80%) and validation (10%) subsets test (10%)
        train_data = output[:train_end_index]
        valid_data = output[train_end_index:val_end_index]
        test_data = output[val_end_index:]

        with open(os.path.join(outLabelPath, "tmh_rgb_train_split_1.txt"), 'w') as f:
            f.write('\n'.join(train_data))
        with open(os.path.join(outLabelPath, "tmh_rgb_val_split_1.txt"), 'w') as f:
            f.write('\n'.join(valid_data))
        with open(os.path.join(outLabelPath, "tmh_rgb_test_split_1.txt"), 'w') as f:
            f.write('\n'.join(test_data))
    

    @staticmethod
    def extract_overlaps(intervals):
        # Expand the intervals into individual events and sort them
        events = sorted((t, startOrEnd, label) for start, end, label in intervals for t, startOrEnd in [(start, -1), (end, 1)])

        interval = []
        labels = set()

        for i in range(1, len(events)):
            t1, startOrEnd1, label1 = events[i - 1]
            t2, _, _ = events[i]

            if startOrEnd1 == -1:
                labels.add(label1)
            elif startOrEnd1 == 1 and label1 in labels:
                labels.remove(label1)

            if t1 != t2 and labels:
                interval.append((t1, t2, ','.join(sorted(labels))))

            newIntervals = []
            prevEnd = 0

            for start, end, label in interval:
                if start > prevEnd:
                    newIntervals.append((prevEnd, start, 'normal'))
                newIntervals.append((start, end, label))
                prevEnd = end
        
        return newIntervals
    

    @staticmethod
    def vid2jpg(videoName, classPath, dstPath, startTime, endTime, semaphore):
        with semaphore:
            videoPath = os.path.join(classPath, videoName)

            cmd = f'ffmpeg -i "{videoPath}" -vf "scale=640:-1"  -crf 30 -ss {startTime} -to {endTime} "{dstPath}"'

            subprocess.call(cmd, shell=True)


    @staticmethod
    def timestamp_to_seconds(timestamp):
        minutes, seconds = map(int, timestamp.split(':'))
        return minutes * 60 + seconds



if __name__ == '__main__':
    tmhDataGenerator = genDataset("/home/comp/cszsliu/tmh_wardvideo/labels-all.xlsx")
    # tmhDataGenerator.cut_videos("/home/comp/cszsliu/tmh_wardvideo", "/home/comp/cszsliu/tmh_wardvideo/imgSide_video")
    tmhDataGenerator.gen_label('/home/comp/cszsliu/tmh_wardvideo/imgSide_video')