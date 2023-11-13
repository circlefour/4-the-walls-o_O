"""
object containing all video streams in a dictionary indexed by name
provides the option to add streams and remove streams
then reading frames actually means something
"""

"""
fuck it i think this shit is stupid
like
either i store a dictionary of them somwhere, wher i can index them by name
and also have the ability to add streams and destroy streams
show streams or not show streams

"""

from imutils.video import WebcamVideoStream

class VideoStreams:
    def __init__(self, sources):
        self.caps = {source: WebcamVideoStream(src=src).start() for source, src in sources.items()}

    def __enter__(self):
        return self.caps
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(self.caps)
        for src, cap in self.caps.items():
            cap.stop()

    # def add_src(self, source):
    #     pass

    # def remove_src(self, src):
    #     pass

    # def get_frame(self, src):
    #     return self.read()

    # def get_all_frames(self):
    #     return {source: cap.read() for source, cap in self.caps.items()}
    
    # def release(self, src):
    #     self.cap.stop()

    # def release_all(self):
    #     pass