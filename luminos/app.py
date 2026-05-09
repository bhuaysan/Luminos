"""Application entry point."""

import sys
from PySide6.QtCore import QtMsgType, qInstallMessageHandler
from PySide6.QtWidgets import QApplication
from luminos.ui.main_window import MainWindow


def _qt_message_handler(msg_type: QtMsgType, context, message: str) -> None:
    # Suppress a benign XCB platform warning that fires on Linux when a menu
    # is dismissed by clicking its title a second time.  The warning originates
    # inside Qt's XCB plugin (QXcbWindow::setMouseGrabEnabled) and is harmless.
    if "grabbing the mouse only for popup" in message:
        return
    if msg_type == QtMsgType.QtWarningMsg:
        print(f"Qt Warning: {message}", file=sys.stderr)
    elif msg_type in (QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
        print(f"Qt Error: {message}", file=sys.stderr)


def main() -> None:
    qInstallMessageHandler(_qt_message_handler)
    app = QApplication(sys.argv)
    app.setApplicationName("Luminos")
    app.setOrganizationName("Luminos")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
