from win10toast import ToastNotifier
from finance_agent.utils.logger import logger
import threading

class AlertManager:
    def __init__(self):
        self.toaster = ToastNotifier()

    def send_desktop_notification(self, title: str, message: str, duration: int = 10):
        """Sends a native Windows 10/11 desktop notification."""
        logger.info(f"[AlertManager] Triggering desktop notification: {title} - {message}")
        try:
            # Run in a separate thread so it doesn't block the main process
            def _show():
                try:
                    self.toaster.show_toast(
                        title=title,
                        msg=message,
                        icon_path=None,  # Optionally add a custom .ico file
                        duration=duration,
                        threaded=False # we are already in a thread
                    )
                except Exception as ex:
                    logger.warning(f"[AlertManager] Windows Toast failed (Headless environment?): {ex}")
            
            thread = threading.Thread(target=_show)
            thread.start()
        except Exception as e:
            logger.error(f"[AlertManager] Failed to start notification thread: {e}")
