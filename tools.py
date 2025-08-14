import math

class WasteClassifier:
    """
    Tools for waste classification, recycling guidance, and disassembly instructions.
    """

    @staticmethod
    def classify_waste_type(item_description: str, material_type: str = None) -> dict:
        """
        Classify waste type and provide recycling category.
        
        Args:
            item_description: Description of the waste item (e.g., "plastic water bottle", "old smartphone")
            material_type: Optional material type hint (e.g., "plastic", "electronic", "metal")
            
        Returns:
            Dict with classification results and basic recycling info
        """
        # Basic classification logic (in real implementation, this would use AI/ML)
        item_lower = item_description.lower()
        
        if any(word in item_lower for word in ["bottle", "container", "plastic", "bag"]):
            category = "Recyclable Plastic"
            instructions = "Check recycling number on bottom. Rinse clean before recycling."
        elif any(word in item_lower for word in ["phone", "computer", "electronic", "battery", "cable"]):
            category = "E-Waste"
            instructions = "Take to certified e-waste facility. May contain valuable materials for recovery."
        elif any(word in item_lower for word in ["glass", "jar", "window"]):
            category = "Glass Recyclable"
            instructions = "Remove lids/caps. Rinse clean. Check local glass recycling guidelines."
        elif any(word in item_lower for word in ["paper", "cardboard", "newspaper", "magazine"]):
            category = "Paper Recyclable"
            instructions = "Keep dry. Remove any plastic components or tape."
        elif any(word in item_lower for word in ["metal", "can", "aluminum", "steel"]):
            category = "Metal Recyclable"
            instructions = "Rinse clean. Remove labels if possible. Check for recycling codes."
        elif any(word in item_lower for word in ["food", "organic", "compost"]):
            category = "Organic Waste"
            instructions = "Suitable for composting if available. Check local composting programs."
        else:
            category = "General Waste"
            instructions = "May require special disposal. Check local waste management guidelines."
        
        return {
            "item": item_description,
            "category": category,
            "recycling_instructions": instructions,
            "material_type": material_type or "Not specified"
        }

    @staticmethod
    def get_disassembly_guidance(item_type: str, safety_level: str = "basic") -> dict:
        """
        Provide disassembly guidance for items before recycling.
        
        Args:
            item_type: Type of item to disassemble (e.g., "smartphone", "laptop", "appliance")
            safety_level: Safety information level ("basic", "detailed", "professional")
            
        Returns:
            Dict with disassembly steps and safety warnings
        """
        item_lower = item_type.lower()
        
        if "phone" in item_lower or "smartphone" in item_lower:
            return {
                "item": item_type,
                "safety_warnings": [
                    "Remove battery if possible - may contain lithium",
                    "Avoid puncturing battery",
                    "Use proper tools to avoid cuts"
                ],
                "disassembly_steps": [
                    "Power off completely",
                    "Remove SIM card and memory card",
                    "Remove back cover carefully",
                    "Disconnect battery connector",
                    "Remove screws and separate components",
                    "Sort materials: plastic, metal, circuit boards"
                ],
                "recyclable_components": ["Battery", "Circuit boards", "Metal frame", "Glass screen", "Plastic housing"]
            }
        elif "laptop" in item_lower or "computer" in item_lower:
            return {
                "item": item_type,
                "safety_warnings": [
                    "Disconnect all power sources",
                    "Ground yourself to prevent static damage",
                    "Be careful of sharp edges"
                ],
                "disassembly_steps": [
                    "Remove battery and unplug all cables",
                    "Remove screws from back panel",
                    "Carefully separate keyboard and screen",
                    "Remove hard drive and RAM",
                    "Disconnect motherboard",
                    "Sort components by material type"
                ],
                "recyclable_components": ["Hard drive", "RAM", "Motherboard", "Battery", "Metal casing", "Cables"]
            }
        else:
            return {
                "item": item_type,
                "safety_warnings": [
                    "Always prioritize safety - use proper protective equipment",
                    "Check for hazardous materials",
                    "Follow local safety guidelines"
                ],
                "disassembly_steps": [
                    "Research item-specific disassembly instructions",
                    "Gather appropriate tools",
                    "Work in well-ventilated area",
                    "Sort components by material type"
                ],
                "recyclable_components": ["Varies by item type - consult recycling guidelines"]
            }

class RecyclingCart:
    """
    Tracks items for proper recycling and disposal.
    """
    # In-memory recycling cart
    _recycling_items = []
    
    @staticmethod
    def add_to_recycling_list(item_name: str, category: str, quantity: int = 1, notes: str = "") -> dict:
        """
        Add an item to the recycling tracking list.
        
        Args:
            item_name: Name/description of the item
            category: Recycling category (e.g., "E-Waste", "Recyclable Plastic")
            quantity: Number of items
            notes: Additional notes or special instructions
            
        Returns:
            Dict with confirmation message and current recycling list
        """
        item = {
            "item_name": item_name,
            "category": category,
            "quantity": quantity,
            "notes": notes,
            "date_added": "Current session"  # In real app, would use actual timestamp
        }
        
        # Check if item already exists
        for existing_item in RecyclingCart._recycling_items:
            if existing_item["item_name"] == item_name and existing_item["category"] == category:
                # Update quantity
                existing_item["quantity"] += quantity
                if notes:
                    existing_item["notes"] = f"{existing_item['notes']}; {notes}" if existing_item["notes"] else notes
                return {
                    "message": f"Updated {item_name} quantity to {existing_item['quantity']} in recycling list",
                    "recycling_list": RecyclingCart._recycling_items
                }
        
        # Add new item
        RecyclingCart._recycling_items.append(item)
        
        return {
            "message": f"Added {quantity} {item_name} ({category}) to your recycling list",
            "recycling_list": RecyclingCart._recycling_items
        }
    
    @staticmethod
    def get_recycling_items() -> list:
        """
        Get all items currently in the recycling list.
        
        Returns:
            List of items in the recycling list with their details
        """
        return RecyclingCart._recycling_items
    
    @staticmethod
    def clear_recycling_list() -> dict:
        """
        Clear all items from the recycling list.
        
        Returns:
            Confirmation message
        """
        RecyclingCart._recycling_items = []
        return {"message": "Recycling list has been cleared"}

    @staticmethod
    def get_disposal_locations(category: str, location: str = "general") -> dict:
        """
        Get disposal/recycling location suggestions for different waste categories.
        
        Args:
            category: Waste category (e.g., "E-Waste", "Hazardous")
            location: Geographic location for local recommendations
            
        Returns:
            Dict with location suggestions and contact info
        """
        locations = {
            "E-Waste": [
                "Best Buy - E-waste recycling program",
                "Local municipal recycling centers",
                "Manufacturer take-back programs",
                "Certified e-waste recycling facilities"
            ],
            "Hazardous": [
                "Municipal hazardous waste collection events",
                "Auto parts stores (for batteries, oil)",
                "Paint stores (for paint disposal)",
                "Specialized hazardous waste facilities"
            ],
            "General Recyclables": [
                "Curbside recycling programs",
                "Local recycling centers",
                "Grocery store recycling bins",
                "Community recycling drop-offs"
            ]
        }
        
        return {
            "category": category,
            "suggested_locations": locations.get(category, locations["General Recyclables"]),
            "note": f"Check local guidelines for {location} area. Contact your waste management provider for specific locations."
        }