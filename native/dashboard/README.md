# Native dashboard

This viewer contains the same route, components, types, styles and layout as the browser-qualified native application. `qualified-ui-manifest.json` pins their hashes.

From the repository root, run `python -B scripts/start_native_dashboard.py`. It reuses the installed root dashboard dependencies through one scoped junction and refuses a conflicting dependency directory or an occupied port. No package installation is performed. The server uses loopback port 4301 and the `/native-stop-v2` route; use the complete link printed by the native launcher to bind a run.

See [setup and release notes](../../docs/NATIVE-RELEASE.md). The TypeScript check passes. The latest repository-entry normal run delivered all 60 original frames and two strict maps, with zero browser JavaScript or decoding errors. The application Stop qualification preserves incomplete coverage and confirms owned cleanup.
