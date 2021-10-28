from visdom import Visdom
import json

class Visualizer(object):
    """Visualizer"""
    def __init__(self, port='13579', env='main', id=None):
        self.vis = Visdom(port=port, env=env)
        self.id = id
        self.env = env

    def vis_scalar(self, name, x, y, opts=None):
        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]

        if self.id is not None:
            name = "[%s]" % self.id + name
        default_opts = {'title': name}
        if opts is not None:
            default_opts.update(opts)

        self.vis.line(X=x, Y=y, win=name, opts=default_opts, update='append')

    def vis_image(self, name, img, env=None, opts=None):
        """vis image in visdom"""
        if env is None:
            env = self.env
        if self.id is not None:
            name = "[%s]" % self.id + name
        
        default_opts = {'title': name}
        if opts is not None:
            default_opts.update(opts)

        self.vis.image(img=img, win=name, opts=default_opts, env=env)

    def vis_table(self, name, tbl, opts=None):
        tbl_str = "<table width=\"100%\"> "
        tbl_str += "<tr> \
                    <th>Term</th> \
                    <th>Value</th> \
                    </tr>"
        for k, v in tbl.items():
            tbl_str += "<tr> \
                        <td>%s</td> \
                        <td>%s</td> \
                        </tr>" % (k, v)

        tbl_str += "</table>"

        default_opts = {'title': name}
        if opts is not None:
            default_opts.update(opts)
        self.vis.text(tbl_str, win=name, opts=default_opts)


if __name__ == '__main__':
    import numpy as np
    import cv2
    from PIL import Image

    vis = Visualizer(port=8097, env='main')
    tbl = {"lr": 214, "momentum": 0.9}
    vis.vis_table("test_table", tbl)
    tbl = {"lr": 244444, "momentum": 0.9, "haha": "hoho"}
    vis.vis_table("test_table", tbl)

    vis.vis_scalar('loss', x=0, y=1.4)
    vis.vis_scalar('loss', x=2, y=4)
    vis.vis_scalar('loss', x=4, y=6)

    img = np.array(Image.open('datasets/demo.png').convert('RGB'))
    img = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3)).transpose(2, 0, 1)
    vis.vis_image('image', img)