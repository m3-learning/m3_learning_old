import glob as glob
# from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import cv2


def make_movie(movie_name, input_folder, output_folder, file_format,
               fps, output_format='mp4', reverse=False):
    """
    Function which makes movies from an image series

    Parameters
    ----------
    movie_name : string
        name of the movie
    input_folder  : string
        folder where the image series is located
    output_folder  : string
        folder where the movie will be saved
    file_format  : string
        sets the format of the files to import
    fps  : numpy, int
        frames per second
    output_format  : string, optional
        sets the format for the output file
        supported types .mp4 and gif
        animated gif create large files
    reverse : bool, optional
        sets if the movie will be one way of there and back
    """

    # searches the folder and finds the files
    file_list = glob.glob(input_folder + '/*.' + file_format)

    # Sorts the files by number makes 2 lists to go forward and back
    list.sort(file_list)
    file_list_rev = glob.glob(input_folder + '/*.' + file_format)
    list.sort(file_list_rev, reverse=True)

    # combines the file list if including the reverse
    if reverse:
        new_list = file_list + file_list_rev
    else:
        new_list = file_list

    print(new_list)

    # Create the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        f"{output_folder}/{movie_name}.mp4", fourcc, fps, cv2.imread(new_list[0]).shape[0:2])

    print(f"{output_folder}{movie_name}")
    # Add frames to the video
    for image in new_list:
        frame = cv2.imread(image)
        video_writer.write(frame)

    video_writer.release()

    # Release the resources
    cv2.destroyAllWindows()
