"""
输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。
要求不能创建任何新的结点，只能调整树中结点指针的指向。
"""
class Tree:
      def __init__(self,val,left = None,right = None,next = None,next2 = None):
          self.val = val
          self.left = left
          self.right = right
          self.next = next
          self.next2 = next2

class build_tree:
      def __init__(self,root = None):
          self.root = root

      def num2tree(self,nums):
          n = len(nums)
          self.root = Tree(nums[0])
          i = 1
          dequeue = [self.root]
          while i < n:
                node = dequeue.pop(0)
                if node:
                   node.left = Tree(nums[i]) if nums[i] is not None else None
                   dequeue.append(node.left)
                   i += 1
                   if i < n:
                      node.right = Tree(nums[i]) if nums[i] is not None else None
                      dequeue.append(node.right)
                   i += 1
          return self.root

class Solution:
      def __init__(self,root):
          self.root = root

      def travel(self):
          answer = []
          dequeue = [self.root]
          while dequeue:
                node = dequeue.pop(0)
                answer.append(node.val)
                if node.left:
                   dequeue.append(node.left)
                if node.right:
                   dequeue.append(node.right)
          return answer

      def gen_link(self,answer):
          self.root.val = answer[0]
          cur_node = self.root
          n = len(answer)
          for i in range(1,n):
              node = Tree(answer[i])
              cur_node.next = node
              node.next2 = cur_node
              cur_node = node
          return self.root

      def pileline(self):
          ans = self.travel()
          ans.sort()
          result = self.gen_link(ans)
          return result

      def travel_root(self):
          cur_node = self.root
          ans = [cur_node.val]
          while  cur_node.next:
                ans.append(cur_node.next.val)
                cur_node = cur_node.next

          res = []
          while  cur_node:
                res.append(cur_node.val)
                cur_node = cur_node.next2
          return ans,res


if __name__ == '__main__':
    nums =[4,6,8,10,12,14,16]
    tree = build_tree()
    tree.num2tree(nums)
    instance = Solution(tree.root)
    instance.pileline()
    ans,res = instance.travel_root()
    print(ans)
    print(res)



