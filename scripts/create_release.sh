#!/bin/bash

# Script to create GitHub release
# Usage: ./scripts/create_release.sh v1.0.0 "Release message"

VERSION=$1
MESSAGE=$2

if [ -z "$VERSION" ] || [ -z "$MESSAGE" ]; then
    echo "Usage: ./create_release.sh <version> <message>"
    echo "Example: ./create_release.sh v1.0.0 'Initial release'"
    exit 1
fi

echo "Creating release $VERSION..."

# Commit changes
git add .
git commit -m "Prepare $VERSION release"

# Create tag
git tag -a $VERSION -m "$MESSAGE"

# Push
git push origin main
git push origin $VERSION

# Create GitHub release
gh release create $VERSION \
  --title "$VERSION - LCNN Brain Stroke Segmentation" \
  --notes "$MESSAGE" \
  --target main

echo "Release $VERSION created successfully!"
echo "Visit: https://github.com/hoangtung386/brain-stroke-segmentation/releases/tag/$VERSION"
