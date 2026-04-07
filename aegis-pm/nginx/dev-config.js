// Aegis PM – dev API override
// Mounted by docker-compose.override.yml so dashboard calls API directly
// instead of using the /api nginx proxy path.
window.AEGIS_API_URL = 'http://localhost:8000';
