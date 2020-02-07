/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package leetcode;

/**
 *
 * @author Lipi
 */
class Max_Subarray_Sum {

    public static void main(String[] args) {
        Max_Subarray_Sum mss = new Max_Subarray_Sum();
        int[] nums = new int[]{-2, 1, -3, 4, -1,0};

        int ans = mss.maxSubArray(nums);
        System.out.println("answer is:"+ans);
    }

    private int maxSubArray(int[] nums) {
        int maxno = nums[0];
        int ans = nums[0];
        if(nums.length == 1){
            return ans;
        }
        if(nums.length == 0){
            return -1;
        }
        for (int i = 1; i < nums.length; i++) {
          int a = nums[i];
          int b = maxno+nums[i];
          System.out.println("a=" + a +" ,b="+b);
          maxno = Math.max(a,b);
          System.out.println("MaxNo"+maxno);
          ans = Math.max(ans,maxno);
          
          System.out.println("ans"+ans);
        }
        return ans;
    }
}
