import cv2

video = cv2.VideoCapture()
video.open('../Files/movingball.mp4')
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
result = cv2.VideoWriter(
    'result.avi',
    fourcc, 20, size)

counter = 1

while True:
    success, frame_rgb = video.read()
    if not success:
        break
    print('klatka {} z {}'.format(counter, total_frames))
    if counter % 2 == 0:
        cv2.putText(frame_rgb, 'pilka', (100, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 4)

    result.write(frame_rgb)
    counter = counter + 1

video.release()
result.release()
