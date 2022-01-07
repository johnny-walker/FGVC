import os
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2

class PgmBase(tk.Frame):
    canvas = None
    cvImage = None
    lblMsg = None

    btnOpen = None
    btnReset = None
    btnPlay = None
    btnPause = None
    btnSnap = None

    mouseLeftDown = False
    imgPosX = 0
    imgPosY = 0

    def __init__(self, root, width=640, height=480):
        super().__init__(root)
        self.root = root
        self.frame = self
        self.tkimage = None

        # configure window
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center = True
        x = (screen_width - width)//2 if center else 0
        y = 50 if center else 0
        self.border = 2
        self.padding = 2
        self.title = 20
        self.msgHeight = self.btnHeight = 20
        self.root.width = width + self.padding*2 + self.border*2
        self.root.height = height + self.msgHeight + self.btnHeight + self.border*2 + self.padding*2+ self.title
        geometry = '{0:d}x{1:d}+{2:d}+{3:d}'.format(root.width, root.height, x, y) 
        root.geometry(geometry)    # ex. root.geometry('600x400+250+50')
        root.resizable(False, False)
        self.root.title('Image Viewer')
        self.imageStartPos = (0, 0)
        self.imageClickPos = (0, 0)
        self.imgResize = (width, height)

        self.loadLayout()
        self.bindBtnEvents()
    
    def changeBtnStyle(self, widget, active):
        btn = None
        if widget == 'brush':
            btn = self.btnBrush
        elif widget == 'brush_add':
            btn = self.btnBrushAdd
        elif widget == 'brush_erase':
            btn = self.btnBrushErase
        elif widget == 'blend':
            btn = self.btnBlend

        if btn is not None:
            if active:
                btn.configure(foreground = 'purple')
            else:
                btn.configure(foreground = 'black')

    def changeCursor(self, style):
        self.canvas.config(cursor = style)

    def bindBtnEvents(self):
        self.root.protocol("WM_DELETE_WINDOW", self.onExit)
        self.root.bind("<Configure>", self.onResize)
        self.root.bind_all('<Key>', self.onKey)                         # pure virtual
        self.btnPrev['command'] = lambda : self.onPrev()                # pure virtual
        self.btnNext['command'] = lambda : self.onNext()                # pure virtual
        self.btnBrush['command'] = lambda : self.onBrush()              # pure virtual
        self.btnBrushAdd['command'] = lambda : self.onBrushAdd()        # pure virtual
        self.btnBrushErase['command'] = lambda : self.onBrushErase()    # pure virtual
        self.btnBlend['command'] = lambda : self.onBlend()              # pure virtual
        self.btnReset['command'] = lambda : self.onReset()              # pure virtual
        self.btnSave['command'] = lambda : self.onSave()                # pure virtual
      
        # mouse events
        self.root.bind('<Motion>', self.mouseMove)              
        self.root.bind("<Button-1>", self.mouseLDown)
        self.root.bind("<ButtonRelease-1>", self.mouseLRelease)
        self.root.bind("<MouseWheel>", self.mouseWheel)

    
    def mouseMove(self, event):
        if event.widget == self.canvas:
            x, y = event.x, event.y
            #print('{}, {}'.format(x, y))
            self.imgPosX = x-self.imageStartPos[0]
            self.imgPosY = y-self.imageStartPos[1]

            # show image pixel location
            msg = '({:d}, {:d})'.format(self.imgPosX, self.imgPosY)
            self.lblMsg['text'] = msg
    
    def mouseLDown(self, event):
        if event.widget == self.canvas:
            self.mouseLeftDown = True
            x, y = event.x, event.y
            self.imageClickPos = (x-self.imageStartPos[0], y-self.imageStartPos[1])
            self.mouseLClick(event)

    def mouseLRelease(self, event):
        self.mouseLeftDown = False

    # virtual func
    def mouseLClick(self, event):
        print('mouseLClick')
    
    # virtual func
    def mouseWheel(self, event):
        print (event.delta)

    def hitTestImageRect(self, event,pt):
        if event.widget == self.canvas:
            x1, y1 = 0, 0
            x2, y2 = x1+self.imgResize[0], y1+self.imgResize[1]
            x, y = pt
            if (x1 < x and x < x2):
                if (y1 < y and y < y2):
                    return True
        return False

    def onResize(self, event):
        if event.widget == self.canvas:
            self.canvas.update()
            self.imgWidth = self.canvas.winfo_width()
            self.imgHeight = self.canvas.winfo_height()

    def onKey(self, event):
        if event.char == event.keysym or len(event.char) == 1:
            if event.keysym == 'space':
                print("Space")             
            elif event.keysym == 'Escape':
                self.root.destroy()

    def onExit(self):
        if messagebox.askyesno("Exit", "Do you want to quit the application?"):
            self.root.destroy()

    def run(self):
        self.root.mainloop()

    def defineLayout(self, widget, cols=1, rows=1):
        for c in range(cols):    
            widget.columnconfigure(c, weight=1)
        for r in range(rows):
            widget.rowconfigure(r, weight=1)
    
    def loadLayout(self):
        align_mode = 'nswe'

        self.imgWidth = self.root.width - self.padding*2 - self.border*2
        self.imgHeight = self.root.height - self.btnHeight - self.msgHeight - self.padding*2 - self.border*2
        self.canvas = tk.Canvas(self.root,  width=self.imgWidth , height=self.imgHeight , bg='gray')
        divBtnArea = tk.Frame(self.root,  width=self.imgWidth , height=self.btnHeight , bg='white')
        divMsg = tk.Frame(self.root,  width=self.imgWidth , height=self.msgHeight , bg='black')

        self.canvas.grid(row=0, column=0, padx=self.padding, pady=self.padding, sticky=align_mode)
        divBtnArea.grid(row=1, column=0, padx=self.padding, pady=self.padding, sticky=align_mode)
        divMsg.grid(row=2, column=0, padx=self.padding, pady=self.padding, sticky=align_mode)

        self.defineLayout(self.root)
        self.defineLayout(self.canvas)
        self.defineLayout(divMsg)

        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        self.btnPrev = tk.Button(divBtnArea, text='prev')
        self.btnPrev.pack(side='left')

        self.btnNext = tk.Button(divBtnArea, text='next')
        self.btnNext.pack(side='left')

        self.btnBrush = tk.Button(divBtnArea, text='brush')
        self.btnBrush.pack(side='left')
        
        self.btnBrushAdd = tk.Button(divBtnArea, text=' + ')
        self.btnBrushAdd.pack(side='left')        
        
        self.btnBrushErase = tk.Button(divBtnArea, text=' - ')
        self.btnBrushErase.pack(side='left')

        self.btnBlend = tk.Button(divBtnArea, text='blend')
        self.btnBlend.pack(side='left')

        self.btnReset = tk.Button(divBtnArea, text='reset')
        self.btnReset.pack(side='left')

        self.btnSave = tk.Button(divBtnArea, text='save')
        self.btnSave.pack(side='left')

        # label as message
        self.lblMsg = tk.Label(divMsg, text='show message here', bg='black', fg='white')
        self.lblMsg.grid(row=0, column=0, sticky='w')

        self.canvas.update()
        self.imgWidth = self.canvas.winfo_width() - self.padding * 2
        self.imgHeight = self.canvas.winfo_height() - self.padding * 5
        print("image size =", self.imgWidth, self.imgHeight)

    def showMessage(self, msg):
        self.lblMsg['text'] = msg
        
    # virtual func
    def onPrev(self):
        print('onPrev')
    
    # virtual func
    def onNext(self):
        print('onNext')
    
    # virtual func
    def onBrush(self):
        print('onBrush')

    # virtual func
    def onBrushAdd(self):
        print('onBrushAdd')

    # virtual func
    def onBrushErase(self):
        print('onBrushErase')

    # virtual func
    def onBlend(self):
        print('onBlend')

    def loadImage(self, path):
        img = cv2.imread(path)
        im = self.resize(img)
        self.showMessage("file {0:s} loaded".format(path))
        return im

    # img : cv image
    def updateImage(self, img, forceCreate=False):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        self.tkimage = ImageTk.PhotoImage(im)

        if forceCreate or not self.cvImage:
            if self.cvImage:
                self.canvas.delete(self.cvImage)
            self.cvImage = self.canvas.create_image(self.imageStartPos, image=self.tkimage, anchor = 'nw')
        else:
            self.canvas.itemconfig(self.cvImage, image = self.tkimage)

    
    def resize(self, img):
        self.imgResize = self.dimResize(img)
        return cv2.resize(img, self.imgResize)
    
    def dimResize(self, im):
        tar_ratio = self.imgHeight / self.imgWidth
        im_ratio = im.shape[0] / im.shape[1]
        if tar_ratio > im_ratio:
            # scale by width
            width = self.imgWidth
            height = round(width * im_ratio)
        else:
            # scale by height
            height = self.imgHeight
            width = round(height / im_ratio)

        X = (self.imgWidth  - width )//2 + self.padding*2  
        Y = (self.imgHeight - height)//2 + self.padding*2
        self.imageStartPos = (X, Y)
        #print(self.imageStartPos)
        return (width, height)

if __name__ == '__main__':
    program = PgmBase(tk.Tk(), width=800, height=600)
    program.loadLayout()
    program.bindBtnEvents()

    # load image data 
    cwd = os.getcwd()
    tiger = os.path.join(cwd, "data/tiger.jpeg")
    program.loadImage(tiger)
    program.run()
