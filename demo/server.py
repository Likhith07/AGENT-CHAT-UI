import os

# Import components from the server_bundle
from server_bundle.app_setup import app, logger
from server_bundle.routes import main_routes
from server_bundle.state_management import sync_threads_and_sessions

# Register the blueprint with the app
app.register_blueprint(main_routes)

def initialize_app_state_on_startup():
    logger.info("Initializing application state on startup...")
    sync_threads_and_sessions()
    logger.info("Application state initialized on startup.")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 2024))
    print(f"Server running at http://localhost:{port}")
    initialize_app_state_on_startup()
    app.run(host='0.0.0.0', port=port, debug=True) 