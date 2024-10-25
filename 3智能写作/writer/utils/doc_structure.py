class Node:
        def __init__(self, name='',instruction = '', content = ''):
            self.name = name
            self.instruction = instruction
            self.content = content
            self.subheading = {}


class TitleHierarchy:
    def __init__(self):
        self.root = Node()
    

    def add_recursive_titles(self, titles):
        """
        Add recursive titles to the hierarchy.

        param titles: A list of dictionaries representing the hierarchical structure of titles.
        """
        self._add_recursive_title(titles, self.root)

    def _add_recursive_title(self, title_info, parent_node):
        """
        Recursively add titles and subheadings to the hierarchy.

        param title_info: A dictionary containing 'task_id', 'instruction', and 'subheadings'.
        param parent_node: The parent Node to which the title should be added.
        """
        task_id = title_info['task_id']
        chapter_name = title_info['chapter_name']
        instruction = title_info['instruction']
        subheadings = title_info['subheadings']
        # Create a new node for the current title
        title_node = Node(name = chapter_name,instruction = instruction)
        parent_node.subheading[task_id] = title_node

        # Recursively add subheadings
        for subheading in subheadings:
            self._add_recursive_title(subheading, title_node)
            
    def get_child_content_by_prefix(self, prefix):
        subheading = self.get_chapter_obj_by_id(prefix).subheading
        all_content = []
        if subheading:
            for chapter_id, obj in subheading.items():
                all_content.append(obj.content) 
        return all_content
         
    def get_subheadings_by_prefix(self, prefix):
        """
        Get subheadings based on the given prefix.

        param root: The root Node of the hierarchical structure representing the relationship between titles and subtitles.
        param prefix: The prefix string to filter the subheadings.
        return: A list of subheading names that match the given prefix.
        """
        # Split the prefix into its components
        prefix_parts = prefix.split('.')
        current_prefix = ""
        # Navigate through the structure using the prefix
        current = self.root
        for part in prefix_parts:
            current_prefix += part
            if current_prefix in current.subheading:
                current = current.subheading[current_prefix]
                current_prefix += '.'
            else:
                # If any part of the prefix is not found, return an empty list
                return []

        # Collect all subheading names at this level
        subheadings = []
        for key, value in current.subheading.items():
            subheadings.append(f"{key} {value.name}")

        return subheadings
    
    def set_obj_by_id(self, path: str):
        """
        Set the content of a node identified by its path.

        param path: A string representing the path to the node, formatted as "x.y.z...".
        param content: The content to be set for the node.
        """
        # Split the path into its components
        path_parts = path.split('.')
        cur_parts = ''
        # Navigate through the structure using the path
        current = self.root
        for part in path_parts:
            cur_parts += part
            if cur_parts in current.subheading:
                current = current.subheading[cur_parts]
                cur_parts += '.'
            else:
                # If any part of the path does not exist, return without setting content
                return
        return current
    

    def set_content_by_id(self, path: str, content: str):
        current = self.set_obj_by_id(path)
        current.content = content 
    
    def set_instruction_by_id(self, path: str, content: str):
        current = self.set_obj_by_id(path)
        current.instruction = content     
        
        
    def set_content_by_headings(self, heading_context_maps):
        """
        Set the contents of nodes identified by their paths.

        param titles: A list of title strings representing the paths.
        param contents: A list of contents corresponding to the titles.
        """
        for title, content in heading_context_maps.items():
            path, _  =  title.split(' ')
            self.set_content_by_id(path, content)


    
    def get_chapter_obj_by_id(self, id: str) -> str:
        """
        Get the chapter name by its hierarchical ID.

        param id: The hierarchical ID of the chapter to look up, formatted as "x.y.z...".
        return: The name of the chapter if found, otherwise an empty string.
        """
        # Split the hierarchical ID into its components
        id_parts = id.split('.')
        cur_parts = ''
        # Start from the root and navigate through the structure using the ID
        current = self.root
        for part in id_parts:
            cur_parts += part
            if cur_parts in current.subheading:
                current = current.subheading[cur_parts]
                cur_parts += '.'
            else:
                # If any part of the ID is not found, return an empty string
                return ''
        # Return the name of the chapter node
        return current
    
    def traverse_and_output(self, node=None,  level=0):
        if node is None:
            node = self.root

        output = []
        for key, child in node.subheading.items():
            title = f"{key} {child.name}"
            output.append((title, child.content, level + 1))
            output.extend(self.traverse_and_output(child, level + 1))

        return output