import ctypes
from pathlib import Path
from time import monotonic, sleep
import winreg

FIELDS = ("size", "flags", "x", "y", "z", "r", "u", "v",
          "buttons", "button_number", "pov", "reserved1", "reserved2")

class JoystickState(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in FIELDS]

def read_controller(slot):
    state = JoystickState()
    state.size = ctypes.sizeof(state)
    state.flags = 0xFF
    if ctypes.windll.winmm.joyGetPosEx(slot, ctypes.byref(state)) == 0:
        return state

def _steam_sdl():
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as key:
            steam_path = Path(winreg.QueryValueEx(key, "SteamPath")[0])
    except OSError:
        steam_path = Path(r"C:\Program Files (x86)\Steam")

    try:
        sdl = ctypes.CDLL(str(steam_path / "SDL3.dll"))
        signatures = {
            "SDL_SetHint": ((ctypes.c_char_p, ctypes.c_char_p), ctypes.c_bool),
            "SDL_Init": ((ctypes.c_uint32,), ctypes.c_bool),
            "SDL_GetGamepads": ((ctypes.POINTER(ctypes.c_int),), ctypes.POINTER(ctypes.c_uint32)),
            "SDL_OpenGamepad": ((ctypes.c_uint32,), ctypes.c_void_p),
            "SDL_CloseGamepad": ((ctypes.c_void_p,), None),
            "SDL_GetGamepadName": ((ctypes.c_void_p,), ctypes.c_char_p),
            "SDL_GetGamepadAxis": ((ctypes.c_void_p, ctypes.c_int), ctypes.c_int16),
            "SDL_GetGamepadButton": ((ctypes.c_void_p, ctypes.c_int), ctypes.c_bool),
            "SDL_UpdateGamepads": ((), None),
            "SDL_Quit": ((), None),
            "SDL_free": ((ctypes.c_void_p,), None),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(sdl, name)
            function.argtypes = arguments
            function.restype = result
    except (OSError, AttributeError):
        return None

    sdl.SDL_SetHint(b"SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", b"1")
    return sdl if sdl.SDL_Init(0x00002000) else None


def _calibrate_sdl(sdl):
    count = ctypes.c_int()
    ids = sdl.SDL_GetGamepads(ctypes.byref(count))
    gamepads = [pad for i in range(count.value) if (pad := sdl.SDL_OpenGamepad(ids[i]))]
    sdl.SDL_free(ids)
    if not gamepads:
        sdl.SDL_Quit()
        return None

    def read_controls(pad):
        triggers = [sdl.SDL_GetGamepadAxis(pad, axis) for axis in (4, 5)]
        buttons = [sdl.SDL_GetGamepadButton(pad, button) for button in range(15)]
        return tuple(triggers + buttons)

    selected = None
    try:
        names = ", ".join((sdl.SDL_GetGamepadName(pad) or b"Gamepad").decode(errors="replace")
                          for pad in gamepads)
        print(f"Controller detected: {names}", flush=True)
        while True:
            input("Release both triggers and all buttons, then press Enter. ")
            sdl.SDL_UpdateGamepads()
            baseline = {pad: read_controls(pad) for pad in gamepads}
            print("Hold L2 by itself, or hold your assigned aim button. Keep R2 released.", flush=True)
            deadline = monotonic() + 15
            while monotonic() < deadline:
                sdl.SDL_UpdateGamepads()
                for pad in gamepads:
                    controls = read_controls(pad)
                    pressed = [index for index, value in enumerate(controls)
                               if value - baseline[pad][index] >= (7000 if index < 2 else 1)]
                    if len(pressed) != 1:
                        continue
                    index = pressed[0]
                    neutral = baseline[pad][index]
                    selected = pad
                    name = ("L2 (left trigger)", "R2 (right trigger)")[index] if index < 2 else "Aim button"
                    print(f"{name} detected. Release it to stop aiming.", flush=True)

                    def held():
                        sdl.SDL_UpdateGamepads()
                        if index < 2:
                            return sdl.SDL_GetGamepadAxis(pad, index + 4) - neutral >= 3500
                        return bool(sdl.SDL_GetGamepadButton(pad, index - 2))

                    return held
                sleep(0.02)
            print("No single aim control registered. Try again or press Ctrl+C to exit.", flush=True)
    finally:
        for pad in gamepads:
            if pad != selected:
                sdl.SDL_CloseGamepad(pad)
        if selected is None:
            sdl.SDL_Quit()


def _calibrate_winmm():
    slots = range(ctypes.windll.winmm.joyGetNumDevs())
    connected = [slot for slot in slots if read_controller(slot) is not None]
    if not connected:
        raise SystemExit("No controller detected. Connect it and launch again.")
    input("Release the controller controls, then press Enter on the keyboard. ")
    baseline = {slot: read_controller(slot) for slot in connected}
    print("Hold L2 by itself, or hold your assigned aim button. Keep R2 released and the sticks still.")
    while True:
        for slot, neutral in baseline.items():
            state = read_controller(slot)
            if state is None or neutral is None:
                continue
            pressed = state.buttons & ~neutral.buttons
            if pressed:
                mask = pressed & -pressed
                print("Aim button detected. Release it to stop aiming.")
                return lambda: bool((read_controller(slot) or JoystickState()).buttons & mask)
            axes = ("x", "y", "z", "r", "u", "v")
            axis = max(axes, key=lambda name: abs(getattr(state, name) - getattr(neutral, name)))
            center = getattr(neutral, axis)
            delta = getattr(state, axis) - center
            if abs(delta) >= 13000:
                direction = 1 if delta > 0 else -1
                def held():
                    current = read_controller(slot)
                    return current is not None and direction * (getattr(current, axis) - center) >= 6500
                print("Aim trigger detected. Release it to stop aiming.")
                return held
        sleep(0.02)


def calibrate():
    sdl = _steam_sdl()
    if sdl is not None:
        held = _calibrate_sdl(sdl)
        if held is not None:
            return held
    return _calibrate_winmm()
