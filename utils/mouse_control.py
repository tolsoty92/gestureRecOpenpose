import pyautogui

class Mouse():

    def __init__(self, x_start=None, y_start=None, x_finish=None, y_finish=None):
        self.x_start, self.y_start = x_start, y_start
        self.x_finish, self.y_finish = x_finish, y_finish
        pyautogui.FAILSAFE = False
        self.palm_was = False

    def move_to(self, x, y, duration=0.3):
        pyautogui.moveTo(x, y, duration=duration)

    def action(self, gesture, x, y):
        if gesture == 'Empty':
            pass
        elif gesture == 'palm':
            self.x_start, self.y_start = x, y
            self.x_finish, self.y_finish = None, None
            self.palm_was = True
        elif gesture == 'fist' and self.palm_was:
            self.x_finish, self.y_finish = x, y

            x_mouse, y_mouse = pyautogui.position()
            x_dest = x_mouse + (self.x_finish - self.x_start)//5
            y_dest = y_mouse + (self.y_finish - self.y_start)//5

            self.move_to(x_dest, y_dest, 0.2)

        elif gesture == '1finger' and self.palm_was:
            pyautogui.click(pyautogui.position())
        elif gesture == '3fingers' and self.palm_was:
            pyautogui.rightClick(pyautogui.position())
        elif gesture == '2fingers' and self.palm_was:
            pyautogui.click(pyautogui.position())
            pyautogui.click(pyautogui.position())

