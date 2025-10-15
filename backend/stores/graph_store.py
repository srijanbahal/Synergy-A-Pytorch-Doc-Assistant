"""
Neo4j graph database implementation for PyTorch knowledge graph.
"""
from typing import Dict, List, Optional, Tuple, Any
from neo4j import GraphDatabase
import structlog
from backend.config import settings

logger = structlog.get_logger(__name__)


class PyTorchGraphStore:
    """Neo4j-based graph store for PyTorch knowledge graph."""
    
    def __init__(self):
        self.uri = settings.neo4j_uri
        self.username = settings.neo4j_username
        self.password = settings.neo4j_password
        self.driver = None
        
        # Connect to Neo4j
        self._connect()
        
        # Initialize schema
        self._create_constraints()
        
        logger.info("Graph store initialized")
    
    def _connect(self):
        """Connect to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def _create_constraints(self):
        """Create unique constraints for nodes."""
        constraints = [
            "CREATE CONSTRAINT module_name IF NOT EXISTS FOR (m:Module) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT class_name IF NOT EXISTS FOR (c:Class) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT function_name IF NOT EXISTS FOR (f:Function) REQUIRE f.name IS UNIQUE",
            "CREATE CONSTRAINT parameter_name IF NOT EXISTS FOR (p:Parameter) REQUIRE p.name IS UNIQUE",
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.debug(f"Constraint may already exist: {e}")
        
        logger.info("Graph constraints created")
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def add_module(self, name: str, description: str = "", url: str = "") -> None:
        """Add a PyTorch module to the graph."""
        query = """
        MERGE (m:Module {name: $name})
        SET m.description = $description,
            m.url = $url,
            m.updated_at = datetime()
        """
        
        with self.driver.session() as session:
            session.run(query, name=name, description=description, url=url)
        
        logger.debug(f"Added module: {name}")
    
    def add_class(self, name: str, module_name: str, description: str = "", 
                  url: str = "", base_classes: List[str] = None) -> None:
        """Add a PyTorch class to the graph."""
        # Add module first
        self.add_module(module_name)
        
        # Add class
        query = """
        MERGE (c:Class {name: $name})
        SET c.description = $description,
            c.url = $url,
            c.updated_at = datetime()
        
        WITH c
        MATCH (m:Module {name: $module_name})
        MERGE (m)-[:CONTAINS]->(c)
        """
        
        with self.driver.session() as session:
            session.run(query, name=name, module_name=module_name, 
                       description=description, url=url)
        
        # Add inheritance relationships
        if base_classes:
            for base_class in base_classes:
                inheritance_query = """
                MATCH (c:Class {name: $class_name})
                MATCH (bc:Class {name: $base_class})
                MERGE (c)-[:INHERITS_FROM]->(bc)
                """
                session.run(inheritance_query, class_name=name, base_class=base_class)
        
        logger.debug(f"Added class: {name} in module {module_name}")
    
    def add_function(self, name: str, module_name: str, class_name: str = None,
                    description: str = "", url: str = "", 
                    parameters: List[Dict] = None, return_type: str = "") -> None:
        """Add a PyTorch function to the graph."""
        # Add module first
        self.add_module(module_name)
        
        # Add function
        query = """
        MERGE (f:Function {name: $name})
        SET f.description = $description,
            f.url = $url,
            f.return_type = $return_type,
            f.updated_at = datetime()
        
        WITH f
        MATCH (m:Module {name: $module_name})
        MERGE (m)-[:CONTAINS]->(f)
        """
        
        with self.driver.session() as session:
            session.run(query, name=name, module_name=module_name, 
                       description=description, url=url, return_type=return_type)
            
            # Add relationship to class if specified
            if class_name:
                class_rel_query = """
                MATCH (f:Function {name: $function_name})
                MATCH (c:Class {name: $class_name})
                MERGE (c)-[:CONTAINS]->(f)
                """
                session.run(class_rel_query, function_name=name, class_name=class_name)
            
            # Add parameters
            if parameters:
                for param in parameters:
                    param_query = """
                    MERGE (p:Parameter {name: $param_name})
                    SET p.type = $param_type,
                        p.default_value = $default_value,
                        p.description = $param_description
                    
                    WITH p
                    MATCH (f:Function {name: $function_name})
                    MERGE (f)-[:HAS_PARAMETER]->(p)
                    """
                    session.run(param_query, 
                              param_name=param.get('name', ''),
                              param_type=param.get('type', ''),
                              default_value=param.get('default_value', ''),
                              param_description=param.get('description', ''),
                              function_name=name)
        
        logger.debug(f"Added function: {name} in module {module_name}")
    
    def add_relationship(self, from_node: str, to_node: str, rel_type: str,
                        from_type: str = "Function", to_type: str = "Function") -> None:
        """Add a relationship between two nodes."""
        query = f"""
        MATCH (from:{from_type} {{name: $from_node}})
        MATCH (to:{to_type} {{name: $to_node}})
        MERGE (from)-[:{rel_type}]->(to)
        """
        
        with self.driver.session() as session:
            session.run(query, from_node=from_node, to_node=to_node)
        
        logger.debug(f"Added relationship: {from_node} -[{rel_type}]-> {to_node}")
    
    def find_related_entities(self, entity_name: str, entity_type: str = "Function",
                             max_depth: int = 2) -> List[Dict]:
        """Find entities related to a given entity within max_depth."""
        query = f"""
        MATCH path = (start:{entity_type} {{name: $entity_name}})-[*1..{max_depth}]-(related)
        WHERE start <> related
        RETURN DISTINCT
            labels(related)[0] as entity_type,
            related.name as entity_name,
            related.description as description,
            length(path) as distance
        ORDER BY distance, entity_name
        LIMIT 50
        """
        
        with self.driver.session() as session:
            result = session.run(query, entity_name=entity_name)
            return [record.data() for record in result]
    
    def get_entity_context(self, entity_name: str, entity_type: str = "Function") -> Dict:
        """Get comprehensive context for an entity."""
        # Get entity details
        entity_query = f"""
        MATCH (e:{entity_type} {{name: $entity_name}})
        RETURN e.name as name, e.description as description, 
               e.url as url, e.return_type as return_type
        """
        
        # Get related entities
        related_query = f"""
        MATCH (e:{entity_type} {{name: $entity_name}})
        OPTIONAL MATCH (e)-[r]-(related)
        RETURN DISTINCT
            type(r) as relationship_type,
            labels(related)[0] as related_type,
            related.name as related_name,
            related.description as related_description
        """
        
        # Get parameters if it's a function
        params_query = f"""
        MATCH (f:Function {{name: $entity_name}})-[:HAS_PARAMETER]->(p:Parameter)
        RETURN p.name as param_name, p.type as param_type, 
               p.default_value as default_value, p.description as param_description
        ORDER BY p.name
        """
        
        with self.driver.session() as session:
            # Get entity details
            entity_result = session.run(entity_query, entity_name=entity_name)
            entity_data = entity_result.single()
            
            if not entity_data:
                return {}
            
            context = dict(entity_data)
            
            # Get related entities
            related_result = session.run(related_query, entity_name=entity_name)
            context['related_entities'] = [record.data() for record in related_result]
            
            # Get parameters if function
            if entity_type == "Function":
                params_result = session.run(params_query, entity_name=entity_name)
                context['parameters'] = [record.data() for record in params_result]
            
            return context
    
    def search_entities(self, search_term: str, entity_types: List[str] = None) -> List[Dict]:
        """Search for entities by name or description."""
        if entity_types is None:
            entity_types = ["Module", "Class", "Function"]
        
        entity_types_str = "|".join(entity_types)
        
        query = f"""
        MATCH (e)
        WHERE (e:{entity_types_str}) AND 
              (e.name CONTAINS $search_term OR 
               e.description CONTAINS $search_term)
        RETURN 
            labels(e)[0] as entity_type,
            e.name as name,
            e.description as description,
            e.url as url
        ORDER BY 
            CASE 
                WHEN e.name CONTAINS $search_term THEN 1
                ELSE 2
            END,
            e.name
        LIMIT 20
        """
        
        with self.driver.session() as session:
            result = session.run(query, search_term=search_term)
            return [record.data() for record in result]
    
    def get_module_hierarchy(self, module_name: str) -> Dict:
        """Get the complete hierarchy of a module."""
        query = """
        MATCH (m:Module {name: $module_name})
        OPTIONAL MATCH (m)-[:CONTAINS]->(c:Class)
        OPTIONAL MATCH (c)-[:CONTAINS]->(f:Function)
        OPTIONAL MATCH (m)-[:CONTAINS]->(f:Function)
        OPTIONAL MATCH (c)-[:INHERITS_FROM]->(parent:Class)
        
        RETURN 
            m.name as module_name,
            m.description as module_description,
            collect(DISTINCT {
                name: c.name,
                description: c.description,
                parent: parent.name,
                functions: [f.name WHERE f IS NOT NULL]
            }) as classes,
            collect(DISTINCT {
                name: f.name,
                description: f.description
            }) as functions
        """
        
        with self.driver.session() as session:
            result = session.run(query, module_name=module_name)
            record = result.single()
            return dict(record) if record else {}
    
    def get_stats(self) -> Dict:
        """Get statistics about the graph database."""
        stats_query = """
        MATCH (n)
        RETURN 
            labels(n)[0] as label,
            count(n) as count
        ORDER BY count DESC
        """
        
        rel_stats_query = """
        MATCH ()-[r]->()
        RETURN 
            type(r) as relationship_type,
            count(r) as count
        ORDER BY count DESC
        """
        
        with self.driver.session() as session:
            # Get node counts
            node_result = session.run(stats_query)
            node_stats = {record["label"]: record["count"] for record in node_result}
            
            # Get relationship counts
            rel_result = session.run(rel_stats_query)
            rel_stats = {record["relationship_type"]: record["count"] for record in rel_result}
            
            return {
                "nodes": node_stats,
                "relationships": rel_stats,
                "uri": self.uri
            }
    
    def clear_all(self) -> None:
        """Clear all data from the graph database."""
        query = "MATCH (n) DETACH DELETE n"
        
        with self.driver.session() as session:
            session.run(query)
        
        logger.info("Graph database cleared")


def main():
    """Main function for testing the graph store."""
    # Initialize graph store
    graph_store = PyTorchGraphStore()
    
    try:
        # Test adding some sample data
        graph_store.add_module("torch", "Core PyTorch functionality")
        graph_store.add_class("Tensor", "torch", "Multi-dimensional array")
        graph_store.add_function("tensor", "torch", description="Create a tensor")
        
        # Test search
        results = graph_store.search_entities("tensor")
        print(f"Search results: {len(results)} entities")
        for result in results:
            print(f"- {result['entity_type']}: {result['name']}")
        
        # Get stats
        stats = graph_store.get_stats()
        print(f"Graph stats: {stats}")
        
    finally:
        graph_store.close()


if __name__ == "__main__":
    main()
