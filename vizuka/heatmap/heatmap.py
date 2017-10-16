class Heatmap():

    def __init__(self, **kwargs):
        """
        Here you create and instanciate the heatmap
        Any argument may be passed
        """
        self.update_colors(kwargs)

    def update_colors(self, **kwargs):
        """
        Update your heatmap with any argument provided
        self.all_colors must be a dict indexed by x by y
        """
        self.all_colors = {}

    def get_all_colors(self):
        return self.all_colors
