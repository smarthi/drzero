#!/bin/bash

# Dr. Zero Search Agent - Quick Start Script
# ==========================================

set -e

echo "🔍 Dr. Zero Search Agent"
echo "========================"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ docker-compose not found. Please install it first."
    exit 1
fi

echo "📦 Building and starting services..."
echo ""

# Use docker compose (v2) or docker-compose (v1)
if docker compose version &> /dev/null 2>&1; then
    docker compose up --build -d
else
    docker-compose up --build -d
fi

echo ""
echo "⏳ Waiting for services to be ready..."

# Wait for UI to be ready
for i in {1..30}; do
    if curl -s http://localhost:8501 > /dev/null 2>&1; then
        break
    fi
    sleep 2
done

echo ""
echo "✅ Demo ready!"
echo ""
echo "🌐 Open in your browser:"
echo "   http://localhost:8501"
echo ""
echo "📊 Restate Admin UI:"
echo "   http://localhost:9070/ui"
echo ""
echo "🛑 To stop:"
echo "   docker-compose down"
echo ""
