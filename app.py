"""Convenience entry point: ``python app.py`` starts the local demo."""

from webdemo.app import app, main

__all__ = ["app"]

if __name__ == "__main__":
    main()
