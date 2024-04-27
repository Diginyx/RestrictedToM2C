import pyglet
window = pyglet.window.Window(800, 600, "OpenGL Test")
@window.event
def on_draw():
    window.clear()
pyglet.app.run()
