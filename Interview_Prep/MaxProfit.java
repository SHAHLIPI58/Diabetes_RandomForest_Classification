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

// Brute Force Method...
public class MaxProfit {
    public int maxProfit(int[] prices) {
        if(prices.length == 0) {
            return 0;
        }
        
        int buyingPrice = prices[0];
        int maxProfit = 0;
        
        for(int i=1;i<prices.length;i++) {
            
            if(prices[i] < buyingPrice) {
                buyingPrice = prices[i];
            }else {
                int currProfit = prices[i]-buyingPrice;
                maxProfit = (currProfit > maxProfit) ? currProfit : maxProfit;
            }
        }
        
        return maxProfit;
    }

    
    public static void main(String []args){
        MaxProfit s = new MaxProfit();
         int[] stockprice1 = new int[]{7,1,5,3,6,4}; 
         int ans = s.maxProfit(stockprice1);
         System.out.println("max profit,,,: "+ans);
    }
}

