#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Documentation Generator and Server Launcher
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Launch the FastAPI server with OpenAPI documentation.
Includes development and production configurations.
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import uvicorn


def launch_api_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = True,
    log_level: str = "info",
    workers: int = 1
):
    """Launch the FastAPI server."""
    
    # Configure server based on environment
    server_config = {
        "app": "src.web.api_server:app",
        "host": host,
        "port": port,
        "log_level": log_level,
    }
    
    if reload and workers == 1:
        # Development mode
        server_config.update({
            "reload": True,
            "reload_dirs": [str(project_root / "src")],
            "reload_includes": ["*.py"],
        })
        print(f"?? Starting API server in DEVELOPMENT mode")
        print(f"?? OpenAPI documentation: http://{host}:{port}/docs")
        print(f"?? Redoc documentation: http://{host}:{port}/redoc")
        print(f"?? OpenAPI JSON: http://{host}:{port}/openapi.json")
    else:
        # Production mode
        server_config.update({
            "workers": workers,
            "access_log": True,
        })
        print(f"?? Starting API server in PRODUCTION mode")
        print(f"?? Workers: {workers}")
    
    print(f"?? Server: http://{host}:{port}")
    print(f"?? Health check: http://{host}:{port}/health")
    print("-" * 50)
    
    # Start server
    uvicorn.run(**server_config)


def generate_openapi_spec():
    """Generate OpenAPI specification file."""
    try:
        from src.web.api_server import app
        import json
        
        # Generate OpenAPI spec
        openapi_spec = app.openapi()
        
        # Save to file
        docs_dir = project_root / "docs"
        docs_dir.mkdir(exist_ok=True)
        
        spec_file = docs_dir / "openapi.json"
        with open(spec_file, 'w') as f:
            json.dump(openapi_spec, f, indent=2)
        
        print(f"? OpenAPI specification saved to: {spec_file}")
        
        # Generate markdown documentation
        markdown_file = docs_dir / "api_reference.md"
        with open(markdown_file, 'w') as f:
            f.write(generate_api_markdown(openapi_spec))
        
        print(f"?? API documentation saved to: {markdown_file}")
        
    except Exception as e:
        print(f"? Failed to generate OpenAPI spec: {e}")
        return False
    
    return True


def generate_api_markdown(openapi_spec: dict) -> str:
    """Generate markdown documentation from OpenAPI spec."""
    
    info = openapi_spec.get('info', {})
    paths = openapi_spec.get('paths', {})
    
    markdown = f"""# {info.get('title', 'API')} Documentation

{info.get('description', '')}

**Version**: {info.get('version', '1.0.0')}

## Base URL

```
http://localhost:8000
```

## Authentication

This API uses Bearer token authentication for production environments.

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8000/endpoint
```

## Endpoints

"""
    
    for path, methods in paths.items():
        markdown += f"\n### `{path}`\n\n"
        
        for method, details in methods.items():
            method_upper = method.upper()
            summary = details.get('summary', '')
            description = details.get('description', '')
            
            markdown += f"#### {method_upper} {path}\n\n"
            
            if summary:
                markdown += f"**Summary**: {summary}\n\n"
            
            if description:
                markdown += f"{description}\n\n"
            
            # Parameters
            parameters = details.get('parameters', [])
            if parameters:
                markdown += "**Parameters:**\n\n"
                for param in parameters:
                    name = param.get('name')
                    param_type = param.get('schema', {}).get('type', 'string')
                    required = param.get('required', False)
                    description = param.get('description', '')
                    required_text = " (required)" if required else " (optional)"
                    markdown += f"- `{name}` ({param_type}){required_text}: {description}\n"
                markdown += "\n"
            
            # Request body
            request_body = details.get('requestBody')
            if request_body:
                markdown += "**Request Body:**\n\n"
                content = request_body.get('content', {})
                for content_type, schema_info in content.items():
                    markdown += f"Content-Type: `{content_type}`\n\n"
                markdown += "\n"
            
            # Responses
            responses = details.get('responses', {})
            if responses:
                markdown += "**Responses:**\n\n"
                for status_code, response_info in responses.items():
                    description = response_info.get('description', '')
                    markdown += f"- `{status_code}`: {description}\n"
                markdown += "\n"
            
            # Example
            markdown += f"**Example:**\n\n```bash\ncurl -X {method_upper} http://localhost:8000{path}\n```\n\n"
            
            markdown += "---\n\n"
    
    return markdown


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Image Processing API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python api_launcher.py                          # Development mode
  python api_launcher.py --prod                   # Production mode
  python api_launcher.py --generate-docs         # Generate documentation
  python api_launcher.py --host 0.0.0.0 --port 8080  # Custom host/port
"""
    )
    
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    
    parser.add_argument(
        "--prod",
        action="store_true",
        help="Run in production mode (no reload, multiple workers)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes for production (default: 4)"
    )
    
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)"
    )
    
    parser.add_argument(
        "--generate-docs",
        action="store_true",
        help="Generate OpenAPI documentation and exit"
    )
    
    args = parser.parse_args()
    
    if args.generate_docs:
        success = generate_openapi_spec()
        sys.exit(0 if success else 1)
    
    # Launch server
    launch_api_server(
        host=args.host,
        port=args.port,
        reload=not args.prod,
        log_level=args.log_level,
        workers=args.workers if args.prod else 1
    )


if __name__ == "__main__":
    main()