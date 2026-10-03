# Installing Cannotate

For the person who will be annotating. You do not need to know any
programming, and you will not need to type any commands.

Setting up takes about ten minutes, most of it waiting for a download. You only
ever do it once.

---

## macOS

### 1. Put the Cannotate folder somewhere sensible

Your Documents folder is fine. Keep the whole folder together — Cannotate reads
things from inside it, so moving single files out will break it.

### 2. Run the setup

Open the `setup` folder, then **right-click** `install_mac.command` and choose
**Open**. A warning appears saying it is from an unidentified developer; click
**Open** again.

> **Right-click → Open matters.** Double-clicking will just show a warning with
> no way past it. This only applies the first time.

A black Terminal window opens and prints what it is doing. It will say it is
downloading about 250 MB; that is normal, and it is the slow part. Leave it
alone until it says:

```
Done — Cannotate is ready.
```

Then press Return to close the window.

If it instead says **Setup did not finish**, something went wrong. The window
tells you what. Send that text to whoever gave you Cannotate.

### 3. Open Cannotate

**Right-click** `Cannotate.app` and choose **Open**, then **Open** again. From
then on you can just double-click it like any other app.

If macOS says Cannotate **"is damaged and can't be opened"**, it is not damaged
— macOS does this to apps downloaded from the internet. Open the `setup` folder
and right-click → Open `install_mac.command` again; if that does not fix it,
ask whoever gave you Cannotate for the unquarantine command.

---

## Windows

### 1. Put the Cannotate folder somewhere sensible

Your Documents folder is fine. Keep the whole folder together.

### 2. Run the setup

Open the `setup` folder and double-click `install_windows.bat`.

Windows may show a blue **"Windows protected your PC"** box. Click **More
info**, then **Run anyway**. This only happens the first time.

A black window opens and prints what it is doing, including a download of about
250 MB. Leave it alone until it says `Done - Cannotate is ready.`

If it says **Setup did not finish**, send whatever it printed to whoever gave
you Cannotate.

### 3. Open Cannotate

Double-click `Cannotate.vbs`. If you prefer to see the progress messages, use
`Cannotate.bat` instead — it does the same thing with a console window.

---

## Saving your work

**Press the 💾 Save button in the annotation window as you go**, or press
**Cmd-S** (macOS) / **Ctrl-S** (Windows). A small green "Saved" time appears
next to the button.

Your work is also saved when you close the annotation window. But if the
program quits unexpectedly, anything since your last save is lost — so save
every so often, the way you would with any other document.

---

## If something stops working later

Run the setup again — `install_mac.command` or `install_windows.bat`. It
rebuilds everything from scratch, and it is much faster the second time because
the download is already on your machine. This fixes most problems.

It never touches your annotations or your recordings. Those live in your bird
folders, not in the Cannotate folder.

---

## What the setup actually did

In case you or your IT department want to know:

- It created one folder — `~/.cannotate` on macOS, `%LOCALAPPDATA%\Cannotate` on
  Windows — holding a private copy of Python and the packages Cannotate needs.
- It changed nothing else on the machine. Nothing was installed system-wide,
  nothing was added to PATH, and no administrator password was needed.
- To remove it completely, delete that folder and the Cannotate folder.
